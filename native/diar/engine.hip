// Host side of the native diarizer: Nemotron 3 Diarization's encoder, upsampler and classifier for one step.
#define ND_BUILD
#include "nd.h"
#include "nd_kernels.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define HIP_OK(x)                                                                                      \
    do {                                                                                               \
        hipError_t e_ = (x);                                                                           \
        if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
    } while (0)

namespace {

constexpr int kD = 512, kHeads = 8, kHeadDim = 64, kFF = 2048, kHead = 192, kUp = 8, kSpeakers = 8;
constexpr int kMaxRows = 2048;  // a step holds the speaker cache, the FIFO and one chunk: about 541 rows
constexpr float kScale = 0.125f;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
int cdiv(int a, int b) { return (a + b - 1) / b; }

// The whole file from the first tensor on, read in 64 MiB pieces through one host buffer.
void read_to_device(const char* path, uint64_t start, uint64_t bytes, char* dev) {
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error(std::string("cannot open ") + path);
    constexpr uint64_t kStep = 64ull << 20;
    std::vector<char> host((size_t)std::min(bytes, kStep));
    LARGE_INTEGER at;
    at.QuadPart = (LONGLONG)start;
    SetFilePointerEx(f, at, nullptr, FILE_BEGIN);
    for (uint64_t done = 0; done < bytes;) {
        const DWORD want = (DWORD)std::min(kStep, bytes - done);
        DWORD got = 0;
        if (!ReadFile(f, host.data(), want, &got, nullptr) || got != want) {
            CloseHandle(f);
            throw std::runtime_error("short read");
        }
        const hipError_t e = hipMemcpy(dev + done, host.data(), want, hipMemcpyHostToDevice);
        if (e != hipSuccess) {
            CloseHandle(f);
            throw std::runtime_error(std::string("hipMemcpy: ") + hipGetErrorString(e));
        }
        done += want;
    }
    CloseHandle(f);
}

}  // namespace

struct NdEngine {
    hipStream_t s{};
    std::unordered_map<std::string, bf16*> weights;  // bf16 copies of the fp32 checkpoint
    char* arena = nullptr;
    bf16* upsampler = nullptr;  // [1536][3 * 192], tap-major
    float* inv_freq = nullptr;
    int layers = 0;
    struct Layer { bf16 *ln1w, *ln1b, *qw, *kw, *vw, *ow, *ob, *ln2w, *ln2b, *f1w, *f1b, *f2w, *f2b; };
    std::vector<Layer> L;
    std::unordered_map<std::string, std::pair<void*, size_t>> scratch;
    std::string stats;

    bf16* w(const std::string& name) {
        auto it = weights.find(name);
        if (it == weights.end()) throw std::runtime_error("missing tensor " + name);
        return it->second;
    }

    template <class T>
    T* buf(const char* name, size_t count) {
        auto& slot = scratch[name];
        size_t bytes = count * sizeof(T);
        if (slot.second < bytes) {
            HIP_OK(hipStreamSynchronize(s));
            if (slot.first) HIP_OK(hipFree(slot.first));
            HIP_OK(hipMalloc(&slot.first, bytes));
            slot.second = bytes;
        }
        return (T*)slot.first;
    }

    void load(const char* path, const char* const* names, const uint64_t* offsets, int count);
    void forward(const float* embeds, int rows, int valid, float* logits);
};

void NdEngine::load(const char* path, const char* const* names, const uint64_t* offsets, int count) {
    const auto t0 = Clock::now();
    if (layers <= 0 || count <= 0) throw std::runtime_error("no model");
    HIP_OK(hipSetDevice(0));
    HIP_OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    if (const std::string bad = gpu_self_check(s); !bad.empty()) throw std::runtime_error("GPU self-check failed: " + bad);
    uint64_t start = UINT64_MAX;
    for (int i = 0; i < count; i++) start = std::min(start, offsets[i]);
    for (int i = 0; i < count; i++)
        if ((offsets[i] - start) % 16) throw std::runtime_error(std::string("misaligned tensor ") + names[i]);
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error(std::string("cannot open ") + path);
    LARGE_INTEGER size;
    GetFileSizeEx(f, &size);
    CloseHandle(f);
    const uint64_t bytes = (uint64_t)size.QuadPart - start;
    char* fp32 = nullptr;
    HIP_OK(hipMalloc(&fp32, bytes));
    read_to_device(path, start, bytes, fp32);
    const double read_ms = ms_since(t0);

    // Every tensor to bf16, each in its own aligned slot of one arena; the fp32 copy is freed after.
    std::vector<uint64_t> ends(offsets, offsets + count);
    std::sort(ends.begin(), ends.end());
    auto tensor_bytes = [&](uint64_t off) {
        const auto next = std::upper_bound(ends.begin(), ends.end(), off);
        return (next == ends.end() ? (uint64_t)size.QuadPart : *next) - off;
    };
    size_t arena_bytes = 0;
    for (int i = 0; i < count; i++) arena_bytes += (tensor_bytes(offsets[i]) / 2 + 255) / 256 * 256;
    HIP_OK(hipMalloc(&arena, arena_bytes));
    char* at = arena;
    for (int i = 0; i < count; i++) {
        const size_t n = tensor_bytes(offsets[i]) / 4;
        nd_to_bf16(s, (const float*)(fp32 + (offsets[i] - start)), n, (bf16*)at);
        weights[names[i]] = (bf16*)at;
        at += (n * 2 + 255) / 256 * 256;
    }
    // The upsampler's conv as an implicit GEMM: its weight goes tap-major, straight from the fp32 copy.
    uint64_t up_off = 0;
    for (int i = 0; i < count; i++)
        if (std::string(names[i]) == "model.upsampler.conv.weight") up_off = offsets[i];
    if (!up_off) throw std::runtime_error("missing tensor model.upsampler.conv.weight");
    HIP_OK(hipMalloc(&upsampler, (size_t)kHead * kUp * 3 * kHead * 2));
    nd_conv_weight(s, (const float*)(fp32 + (up_off - start)), kHead * kUp, kHead, 3 * kHead, upsampler);
    HIP_OK(hipStreamSynchronize(s));
    HIP_OK(hipFree(fp32));

    const std::string P = "model.audio_tower.layers.";
    for (int l = 0; l < layers; l++) {
        const std::string p = P + std::to_string(l) + ".";
        L.push_back(Layer{w(p + "layer_norm1.weight"), w(p + "layer_norm1.bias"), w(p + "self_attn.q_proj.weight"),
                          w(p + "self_attn.k_proj.weight"), w(p + "self_attn.v_proj.weight"),
                          w(p + "self_attn.o_proj.weight"), w(p + "self_attn.o_proj.bias"), w(p + "layer_norm2.weight"),
                          w(p + "layer_norm2.bias"), w(p + "mlp.fc1.weight"), w(p + "mlp.fc1.bias"),
                          w(p + "mlp.fc2.weight"), w(p + "mlp.fc2.bias")});
    }
    // transformers' default rotary frequencies for head_dim 64, theta 10000.
    std::vector<float> inv(32);
    for (int i = 0; i < 32; i++) inv[i] = 1.0f / std::pow(10000.0f, (float)(2 * i) / 64.0f);
    HIP_OK(hipMalloc(&inv_freq, 32 * 4));
    HIP_OK(hipMemcpy(inv_freq, inv.data(), 32 * 4, hipMemcpyHostToDevice));
    char msg[160];
    std::snprintf(msg, sizeof msg, "{\"read_ms\":%.1f,\"load_ms\":%.1f,\"weights_mib\":%.0f}", read_ms, ms_since(t0),
                  arena_bytes / 1048576.0);
    stats = msg;
}

void NdEngine::forward(const float* embeds, int rows, int valid, float* logits) {
    const auto t0 = Clock::now();
    if (rows < 1 || rows > kMaxRows || valid < 1 || valid > rows) throw std::runtime_error("step size out of range");
    float* in = buf<float>("in", (size_t)rows * kD);
    bf16* raw = buf<bf16>("raw", (size_t)rows * kD);
    bf16* x = buf<bf16>("x", (size_t)rows * kD);
    bf16* h = buf<bf16>("h", (size_t)rows * kD);
    bf16* q = buf<bf16>("q", (size_t)rows * kD);
    bf16* k = buf<bf16>("k", (size_t)rows * kD);
    bf16* v = buf<bf16>("v", (size_t)rows * kD);
    bf16* at = buf<bf16>("at", (size_t)rows * kD);
    bf16* ff = buf<bf16>("ff", (size_t)rows * kFF);
    bf16* up = buf<bf16>("up", (size_t)rows * kUp * kHead);
    bf16* dense = buf<bf16>("dense", (size_t)rows * kUp * kHead);
    bf16* out = buf<bf16>("out", (size_t)rows * kUp * kSpeakers);
    float* out32 = buf<float>("out32", (size_t)rows * kUp * kSpeakers);
    std::vector<AttnTile> tiles;
    for (int r = 0; r < rows; r += 16) tiles.push_back({r, std::min(r + 16, rows), 0, valid});
    AttnTile* tiles_d = buf<AttnTile>("tiles", tiles.size());
    HIP_OK(hipMemcpyAsync(tiles_d, tiles.data(), tiles.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(in, embeds, (size_t)rows * kD * 4, hipMemcpyHostToDevice, s));
    nd_to_bf16(s, in, (size_t)rows * kD, raw);

    const std::string A = "model.audio_tower.";
    layer_norm(s, raw, w(A + "input_layer_norm.weight"), w(A + "input_layer_norm.bias"), x, rows, kD);
    for (const Layer& Ly : L) {
        layer_norm(s, x, Ly.ln1w, Ly.ln1b, h, rows, kD);
        Epi e;
        e.out = q; e.ldo = kD;
        gemm(s, h, kD, Ly.qw, rows, kD, kD, e);
        e.out = k;
        gemm(s, h, kD, Ly.kw, rows, kD, kD, e);
        e.out = v;
        gemm(s, h, kD, Ly.vw, rows, kD, kD, e);
        nd_rope(s, q, k, kD, rows, kHeads, inv_freq);
        AttnArgs a{q, kD, kHeadDim, k, kD, kHeadDim, v, kD, kHeadDim, at, kD, kHeadDim, 1, tiles_d, (int)tiles.size(),
                   kHeads, 0, 0, kScale};
        attention(s, a, kHeadDim);
        e = Epi{}; e.bias = Ly.ob; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, at, kD, Ly.ow, rows, kD, kD, e);
        layer_norm(s, x, Ly.ln2w, Ly.ln2b, h, rows, kD);
        e = Epi{}; e.bias = Ly.f1b; e.act = ACT_GELU; e.out = ff; e.ldo = kFF;
        gemm(s, h, kD, Ly.f1w, rows, kFF, kD, e);
        e = Epi{}; e.bias = Ly.f2b; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, ff, kFF, Ly.f2w, rows, kD, kFF, e);
    }
    layer_norm(s, x, w(A + "layer_norm.weight"), w(A + "layer_norm.bias"), h, rows, kD);
    // proj, then the sub-pixel upsampler: its [rows][8 * 192] output is the [rows * 8][192] frames as they lie.
    bf16* proj = q;
    Epi e;
    e.bias = w("model.proj.bias"); e.out = proj; e.ldo = kHead;
    gemm(s, h, kD, w("model.proj.weight"), rows, kHead, kD, e);
    // The classifier opens with a ReLU on these frames, and they feed nothing else.
    e = Epi{}; e.bias = w("model.upsampler.conv.bias"); e.act = ACT_RELU; e.out = up; e.ldo = kUp * kHead;
    conv1d_k3(s, proj, rows, kHead, 1, upsampler, rows, kUp * kHead, 3 * kHead, e);
    e = Epi{}; e.bias = w("classifier.dense.bias"); e.act = ACT_RELU; e.out = dense; e.ldo = kHead;
    gemm(s, up, kHead, w("classifier.dense.weight"), rows * kUp, kHead, kHead, e);
    e = Epi{}; e.bias = w("classifier.out_proj.bias"); e.out = out; e.ldo = kSpeakers;
    gemm(s, dense, kHead, w("classifier.out_proj.weight"), rows * kUp, kSpeakers, kHead, e);
    nd_to_f32(s, out, (size_t)rows * kUp * kSpeakers, out32);
    HIP_OK(hipMemcpyAsync(logits, out32, (size_t)rows * kUp * kSpeakers * 4, hipMemcpyDeviceToHost, s));
    HIP_OK(hipStreamSynchronize(s));
    char msg[96];
    std::snprintf(msg, sizeof msg, "{\"forward_ms\":%.2f,\"rows\":%d,\"valid\":%d}", ms_since(t0), rows, valid);
    stats = msg;
}

extern "C" {

ND_API NdEngine* nd_load(const char* path, const char* const* names, const uint64_t* offsets, int count, int layers,
                         char* err, int err_len) {
    auto* e = new NdEngine();
    try {
        e->layers = layers;
        e->load(path, names, offsets, count);
        return e;
    } catch (const std::exception& x) {
        if (err && err_len > 0) std::snprintf(err, err_len, "%s", x.what());
        nd_free(e);
        return nullptr;
    }
}

ND_API int nd_forward(NdEngine* e, const float* embeds, int rows, int valid, float* logits, char* err, int err_len) {
    try {
        e->forward(embeds, rows, valid, logits);
        return 0;
    } catch (const std::exception& x) {
        if (err && err_len > 0) std::snprintf(err, err_len, "%s", x.what());
        return -1;
    }
}

ND_API const char* nd_stats(NdEngine* e) { return e->stats.c_str(); }

ND_API void nd_free(NdEngine* e) {
    if (!e) return;
    (void)hipDeviceSynchronize();
    for (auto& kv : e->scratch) (void)hipFree(kv.second.first);
    for (void* p : {(void*)e->arena, (void*)e->upsampler, (void*)e->inv_freq}) if (p) (void)hipFree(p);
    if (e->s) (void)hipStreamDestroy(e->s);
    delete e;
}

}  // extern "C"
