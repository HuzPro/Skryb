// Host side of the native CrisperWhisper engine: copies model.safetensors into VRAM and runs audio -> token ids.
#define CW_BUILD
#include "cw.h"
#include "wh_kernels.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define HIP_OK(x)                                                                                      \
    do {                                                                                               \
        hipError_t e_ = (x);                                                                           \
        if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
    } while (0)

namespace {

constexpr int kD = 1280, kHeads = 20, kHeadDim = 64, kFF = 5120;
constexpr int kSamples = 480000, kFrames = 3000, kCtx = 1500, kText = 448;
constexpr int kCrossSplit = 128;  // keys per split in single-row cross-attention
constexpr int kCrossSplits = (kCtx + kCrossSplit - 1) / kCrossSplit;
constexpr float kScale = 0.125f;  // head_dim ** -0.5

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
int cdiv(int a, int b) { return (a + b - 1) / b; }

// Reader threads fill a ring of pageable buffers and the calling thread copies each one to VRAM in order
// (the Qwen3-ASR engine's loader, from a start offset instead of a fixed header).
void stream_to_device(const char* path, uint64_t start, int readers, char* dev, uint64_t bytes) {
    constexpr uint64_t kStep = 16ull << 20;
    constexpr int kSlots = 16;
    const uint64_t chunks = (bytes + kStep - 1) / kStep;
    char* ring = (char*)VirtualAlloc(nullptr, kSlots * kStep, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!ring) throw std::runtime_error("cannot allocate the read ring");
    std::mutex mu;
    std::condition_variable cv;
    std::vector<int64_t> filled(kSlots, -1);
    uint64_t consumed = 0;
    std::string error;
    auto fail = [&](const std::string& why) {
        std::lock_guard<std::mutex> lk(mu);
        if (error.empty()) error = why;
        cv.notify_all();
    };
    auto reader = [&](int r) {
        HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return fail(std::string("cannot open ") + path);
        for (uint64_t c = r; c < chunks; c += readers) {
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return c < consumed + kSlots || !error.empty(); });
                if (!error.empty()) break;
            }
            const int slot = (int)(c % kSlots);
            const uint64_t at = start + c * kStep, want = std::min(kStep, bytes - c * kStep);
            OVERLAPPED ov{};
            ov.Offset = (DWORD)at;
            ov.OffsetHigh = (DWORD)(at >> 32);
            DWORD got = 0;
            if (!ReadFile(h, ring + slot * kStep, (DWORD)want, &got, &ov) || got < want) {
                fail("short read");
                break;
            }
            std::lock_guard<std::mutex> lk(mu);
            filled[slot] = (int64_t)c;
            cv.notify_all();
        }
        CloseHandle(h);
    };
    std::vector<std::thread> pool;
    for (int r = 0; r < readers; r++) pool.emplace_back(reader, r);
    for (uint64_t c = 0; c < chunks; c++) {
        const int slot = (int)(c % kSlots);
        {
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [&] { return filled[slot] == (int64_t)c || !error.empty(); });
            if (!error.empty()) break;
        }
        const hipError_t e = hipMemcpy(dev + c * kStep, ring + slot * kStep, std::min(kStep, bytes - c * kStep),
                                       hipMemcpyHostToDevice);
        if (e != hipSuccess) {
            fail(std::string("hipMemcpy: ") + hipGetErrorString(e));
            break;
        }
        std::lock_guard<std::mutex> lk(mu);
        filled[slot] = -1;
        consumed = c + 1;
        cv.notify_all();
    }
    for (auto& t : pool) t.join();
    VirtualFree(ring, 0, MEM_RELEASE);
    if (!error.empty()) throw std::runtime_error(error);
}

// transformers' mel_filter_bank(201 bins, n_mels, 0-8000 Hz, slaney scale and norm), as [201][n_mels].
std::vector<float> mel_filters(int nm) {
    const double logstep = 27.0 / std::log(6.4);
    auto to_mel = [&](double hz) { return hz >= 1000.0 ? 15.0 + std::log(hz / 1000.0) * logstep : 3.0 * hz / 200.0; };
    auto to_hz = [&](double mel) { return mel >= 15.0 ? 1000.0 * std::exp((mel - 15.0) / logstep) : 200.0 * mel / 3.0; };
    const int nf = 201;
    std::vector<double> ff(nm + 2);
    const double lo = to_mel(0.0), hi = to_mel(8000.0);
    for (int i = 0; i < nm + 2; i++) ff[i] = to_hz(lo + (hi - lo) * i / (nm + 1));
    std::vector<float> out((size_t)nf * nm);
    for (int k = 0; k < nf; k++) {
        const double fft = 8000.0 * k / (nf - 1);
        for (int j = 0; j < nm; j++) {
            const double down = (fft - ff[j]) / (ff[j + 1] - ff[j]);
            const double up = (ff[j + 2] - fft) / (ff[j + 2] - ff[j + 1]);
            out[(size_t)k * nm + j] = (float)(std::max(0.0, std::min(down, up)) * 2.0 / (ff[j + 2] - ff[j]));
        }
    }
    return out;
}

// cos[400], sin[400] of 2*pi*i/400, then the periodic Hann window.
std::vector<float> dft_tables() {
    std::vector<float> t(1200);
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < 400; i++) {
        t[i] = (float)std::cos(2 * pi * i / 400);
        t[400 + i] = (float)std::sin(2 * pi * i / 400);
        t[800 + i] = (float)(0.5 - 0.5 * std::cos(2 * pi * i / 400));
    }
    return t;
}

}  // namespace

struct CwEngine {
    hipStream_t s{};
    char* dev = nullptr;
    uint64_t dev_start = 0;
    std::unordered_map<std::string, uint64_t> table;
    CwConfig cfg{};
    int k1 = 0;  // conv1 K, 3 * n_mels rounded up to 32
    bf16 *conv1 = nullptr, *conv2 = nullptr, *suppress = nullptr;
    float *tables = nullptr, *filters = nullptr;
    unsigned* gmax = nullptr;
    int* ids_dev = nullptr;
    int* host_id = nullptr;
    // One decode step as a graph: the position, the step's causal tile and the picked id stay on the device.
    int *pos_d = nullptr, *last_d = nullptr;
    AttnTile* step_tile = nullptr;
    hipGraphExec_t step_exec = nullptr;
    hipEvent_t step_done = nullptr;
    AttnTile *enc_tiles = nullptr, *row_tile = nullptr, *cross_tiles = nullptr;
    float *part_o = nullptr, *part_ml = nullptr;
    bf16 *xk = nullptr, *xv = nullptr, *kc = nullptr, *vc = nullptr;  // per decoder layer: cross K/V, self K/V cache
    struct Enc { bf16 *ln1w, *ln1b, *qw, *qb, *kw, *vw, *vb, *ow, *ob, *ln2w, *ln2b, *f1w, *f1b, *f2w, *f2b; };
    struct Dec {
        bf16 *ln1w, *ln1b, *qw, *qb, *kw, *vw, *vb, *ow, *ob;
        bf16 *ln2w, *ln2b, *xqw, *xqb, *xkw, *xvw, *xvb, *xow, *xob;
        bf16 *ln3w, *ln3b, *f1w, *f1b, *f2w, *f2b;
    };
    std::vector<Enc> enc;
    std::vector<Dec> dec;
    std::unordered_map<std::string, std::pair<void*, size_t>> scratch;
    std::string stats;
    // CW_PROFILE=1: sync after each decode op and total the host time per op, reported in stats.
    bool profile = std::getenv("CW_PROFILE") != nullptr;
    std::vector<std::pair<std::string, double>> prof;
    Clock::time_point prof_t = Clock::now();
    void tick(const char* tag) {
        if (!profile) return;
        HIP_OK(hipStreamSynchronize(s));
        const double ms = ms_since(prof_t);
        prof_t = Clock::now();
        for (auto& q : prof) if (q.first == tag) { q.second += ms; return; }
        prof.push_back({tag, ms});
    }

    bf16* w(const std::string& name) {
        auto it = table.find(name);
        if (it == table.end()) throw std::runtime_error("missing tensor " + name);
        return (bf16*)(dev + (it->second - dev_start));
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

    void load(const char* path, const char* const* names, const uint64_t* offsets, int count, const int* sup, int n_sup);
    std::vector<int> transcribe(const float* pcm, int n, const int* prompt, int prompt_len);

   private:
    void encoder();
    void decoder_rows(int M, int pos0, const AttnTile* self_tiles, const AttnTile* cross, int n_tiles);
    void step_ops();
    void capture_step();
    void opbench();
    void wait_step();
};

void CwEngine::load(const char* path, const char* const* names, const uint64_t* offsets, int count, const int* sup,
                    int n_sup) {
    const auto t0 = Clock::now();
    if (cfg.n_mels % 8 || cfg.n_mels > 128 || cfg.encoder_layers <= 0 || cfg.decoder_layers <= 0 || cfg.vocab <= 0)
        throw std::runtime_error("unsupported model shape");
    HIP_OK(hipSetDevice(0));
    HIP_OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    dev_start = UINT64_MAX;
    for (int i = 0; i < count; i++) {
        table[names[i]] = offsets[i];
        dev_start = std::min(dev_start, offsets[i]);
        if ((offsets[i] - offsets[0]) % 16) throw std::runtime_error(std::string("misaligned tensor ") + names[i]);
    }
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error(std::string("cannot open ") + path);
    LARGE_INTEGER size;
    GetFileSizeEx(f, &size);
    CloseHandle(f);
    if (count == 0 || dev_start >= (uint64_t)size.QuadPart) throw std::runtime_error("no tensors");
    const uint64_t bytes = (uint64_t)size.QuadPart - dev_start;
    HIP_OK(hipMalloc(&dev, bytes));
    stream_to_device(path, dev_start, 4, dev, bytes);
    const double weights_ms = ms_since(t0);

    const std::string E = "model.encoder.", Dn = "model.decoder.";
    for (int l = 0; l < cfg.encoder_layers; l++) {
        const std::string p = E + "layers." + std::to_string(l) + ".";
        enc.push_back(Enc{w(p + "self_attn_layer_norm.weight"), w(p + "self_attn_layer_norm.bias"),
                          w(p + "self_attn.q_proj.weight"), w(p + "self_attn.q_proj.bias"), w(p + "self_attn.k_proj.weight"),
                          w(p + "self_attn.v_proj.weight"), w(p + "self_attn.v_proj.bias"),
                          w(p + "self_attn.out_proj.weight"), w(p + "self_attn.out_proj.bias"),
                          w(p + "final_layer_norm.weight"), w(p + "final_layer_norm.bias"), w(p + "fc1.weight"),
                          w(p + "fc1.bias"), w(p + "fc2.weight"), w(p + "fc2.bias")});
    }
    for (int l = 0; l < cfg.decoder_layers; l++) {
        const std::string p = Dn + "layers." + std::to_string(l) + ".";
        dec.push_back(Dec{w(p + "self_attn_layer_norm.weight"), w(p + "self_attn_layer_norm.bias"),
                          w(p + "self_attn.q_proj.weight"), w(p + "self_attn.q_proj.bias"), w(p + "self_attn.k_proj.weight"),
                          w(p + "self_attn.v_proj.weight"), w(p + "self_attn.v_proj.bias"),
                          w(p + "self_attn.out_proj.weight"), w(p + "self_attn.out_proj.bias"),
                          w(p + "encoder_attn_layer_norm.weight"), w(p + "encoder_attn_layer_norm.bias"),
                          w(p + "encoder_attn.q_proj.weight"), w(p + "encoder_attn.q_proj.bias"),
                          w(p + "encoder_attn.k_proj.weight"), w(p + "encoder_attn.v_proj.weight"),
                          w(p + "encoder_attn.v_proj.bias"), w(p + "encoder_attn.out_proj.weight"),
                          w(p + "encoder_attn.out_proj.bias"), w(p + "final_layer_norm.weight"),
                          w(p + "final_layer_norm.bias"), w(p + "fc1.weight"), w(p + "fc1.bias"), w(p + "fc2.weight"),
                          w(p + "fc2.bias")});
    }

    // The conv stem as implicit GEMMs: [N][tap][C] weights, conv1's K padded to a multiple of 32.
    k1 = cdiv(3 * cfg.n_mels, 32) * 32;
    HIP_OK(hipMalloc(&conv1, (size_t)kD * k1 * 2));
    HIP_OK(hipMalloc(&conv2, (size_t)kD * 3 * kD * 2));
    wh_conv_weight(s, w(E + "conv1.weight"), kD, cfg.n_mels, k1, conv1);
    wh_conv_weight(s, w(E + "conv2.weight"), kD, kD, 3 * kD, conv2);

    // Suppressed tokens get -inf added to their logit before the argmax, as SuppressTokensLogitsProcessor.
    std::vector<uint16_t> bias(cfg.vocab, 0);
    for (int i = 0; i < n_sup; i++)
        if (sup[i] >= 0 && sup[i] < cfg.vocab) bias[sup[i]] = 0xFF80;
    HIP_OK(hipMalloc(&suppress, (size_t)cfg.vocab * 2));
    HIP_OK(hipMemcpyAsync(suppress, bias.data(), bias.size() * 2, hipMemcpyHostToDevice, s));

    const auto tab = dft_tables();
    const auto fil = mel_filters(cfg.n_mels);
    HIP_OK(hipMalloc(&tables, tab.size() * 4));
    HIP_OK(hipMalloc(&filters, fil.size() * 4));
    HIP_OK(hipMemcpyAsync(tables, tab.data(), tab.size() * 4, hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(filters, fil.data(), fil.size() * 4, hipMemcpyHostToDevice, s));
    HIP_OK(hipMalloc(&gmax, 4));
    HIP_OK(hipMalloc(&ids_dev, kText * 4));
    HIP_OK(hipHostMalloc(&host_id, 4, hipHostMallocDefault));

    // Fixed attention tiles: encoder query blocks of 16 over every key, one decode row over its causal keys,
    // one decode row over the encoder output in splits.
    std::vector<AttnTile> et, ct;
    for (int q = 0; q < kCtx; q += 16) et.push_back({q, std::min(q + 16, kCtx), 0, kCtx});
    for (int sp = 0; sp < kCrossSplits; sp++) ct.push_back({0, 1, sp * kCrossSplit, std::min(kCtx, (sp + 1) * kCrossSplit)});
    const AttnTile rt{0, 1, 0, INT_MAX};
    HIP_OK(hipMalloc(&enc_tiles, et.size() * sizeof(AttnTile)));
    HIP_OK(hipMalloc(&cross_tiles, ct.size() * sizeof(AttnTile)));
    HIP_OK(hipMalloc(&row_tile, sizeof(AttnTile)));
    HIP_OK(hipMemcpyAsync(enc_tiles, et.data(), et.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(cross_tiles, ct.data(), ct.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(row_tile, &rt, sizeof(AttnTile), hipMemcpyHostToDevice, s));
    HIP_OK(hipMalloc(&part_o, (size_t)kCrossSplits * kHeads * 16 * kHeadDim * 4));
    HIP_OK(hipMalloc(&part_ml, (size_t)kCrossSplits * kHeads * 16 * 2 * 4));

    const size_t layers = cfg.decoder_layers;
    HIP_OK(hipMalloc(&xk, layers * kCtx * kD * 2));
    HIP_OK(hipMalloc(&xv, layers * kCtx * kD * 2));
    HIP_OK(hipMalloc(&kc, layers * kText * kD * 2));
    HIP_OK(hipMalloc(&vc, layers * kText * kD * 2));
    HIP_OK(hipMalloc(&pos_d, 4));
    HIP_OK(hipMalloc(&last_d, 4));
    HIP_OK(hipMalloc(&step_tile, sizeof(AttnTile)));
    capture_step();
    HIP_OK(hipEventCreateWithFlags(&step_done, hipEventDisableTiming));
    HIP_OK(hipStreamSynchronize(s));
    if (std::getenv("CW_OPBENCH")) opbench();
    char msg[160];
    std::snprintf(msg, sizeof msg, "{\"weights_ms\":%.1f,\"load_ms\":%.1f,\"weights_gib\":%.2f}", weights_ms, ms_since(t0),
                  bytes / 1073741824.0);
    stats = msg;
}

// Mel, conv stem, encoder layers, then every decoder layer's cross-attention K and V.
void CwEngine::encoder() {
    float* pcm = buf<float>("pcm", kSamples);
    float* mel = buf<float>("mel", (size_t)kFrames * cfg.n_mels);
    bf16* melc = buf<bf16>("melc", (size_t)kFrames * cfg.n_mels);
    wh_log_mel(s, pcm, kSamples, kFrames, cfg.n_mels, tables, filters, mel, gmax, melc);
    bf16* c1 = buf<bf16>("c1", (size_t)kFrames * kD);
    bf16* x = buf<bf16>("ex", (size_t)kCtx * kD);
    const std::string E = "model.encoder.";
    Epi e;
    e.bias = w(E + "conv1.bias"); e.act = ACT_GELU; e.out = c1; e.ldo = kD;
    conv1d_k3(s, melc, kFrames, cfg.n_mels, 1, conv1, kFrames, kD, k1, e);
    e = Epi{}; e.bias = w(E + "conv2.bias"); e.act = ACT_GELU; e.res = w(E + "embed_positions.weight"); e.res_ld = kD;
    e.out = x; e.ldo = kD;
    conv1d_k3(s, c1, kFrames, kD, 2, conv2, kCtx, kD, 3 * kD, e);

    bf16* h = buf<bf16>("eh", (size_t)kCtx * kD);
    bf16* q = buf<bf16>("eq", (size_t)kCtx * kD);
    bf16* k = buf<bf16>("ek", (size_t)kCtx * kD);
    bf16* v = buf<bf16>("ev", (size_t)kCtx * kD);
    bf16* at = buf<bf16>("eat", (size_t)kCtx * kD);
    bf16* ff = buf<bf16>("eff", (size_t)kCtx * kFF);
    const int n_tiles = cdiv(kCtx, 16);
    for (const Enc& L : enc) {
        layer_norm(s, x, L.ln1w, L.ln1b, h, kCtx, kD);
        e = Epi{}; e.bias = L.qb; e.out = q; e.ldo = kD;
        gemm(s, h, kD, L.qw, kCtx, kD, kD, e);
        e = Epi{}; e.out = k; e.ldo = kD;
        gemm(s, h, kD, L.kw, kCtx, kD, kD, e);
        e = Epi{}; e.bias = L.vb; e.out = v; e.ldo = kD;
        gemm(s, h, kD, L.vw, kCtx, kD, kD, e);
        AttnArgs a{q, kD, kHeadDim, k, kD, kHeadDim, v, kD, kHeadDim, at, kD, kHeadDim, 1, enc_tiles, n_tiles, kHeads, 0, 0, kScale};
        attention(s, a, kHeadDim);
        e = Epi{}; e.bias = L.ob; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, at, kD, L.ow, kCtx, kD, kD, e);
        layer_norm(s, x, L.ln2w, L.ln2b, h, kCtx, kD);
        e = Epi{}; e.bias = L.f1b; e.act = ACT_GELU; e.out = ff; e.ldo = kFF;
        gemm(s, h, kD, L.f1w, kCtx, kFF, kD, e);
        e = Epi{}; e.bias = L.f2b; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, ff, kFF, L.f2w, kCtx, kD, kFF, e);
    }
    layer_norm(s, x, w(E + "layer_norm.weight"), w(E + "layer_norm.bias"), h, kCtx, kD);
    for (int l = 0; l < cfg.decoder_layers; l++) {
        e = Epi{}; e.out = xk + (size_t)l * kCtx * kD; e.ldo = kD;
        gemm(s, h, kD, dec[l].xkw, kCtx, kD, kD, e);
        e = Epi{}; e.bias = dec[l].xvb; e.out = xv + (size_t)l * kCtx * kD; e.ldo = kD;
        gemm(s, h, kD, dec[l].xvw, kCtx, kD, kD, e);
    }
}

// M decoder rows at positions pos0.. through every layer, in place in "dx"; their keys and values join the cache.
void CwEngine::decoder_rows(int M, int pos0, const AttnTile* self_tiles, const AttnTile* cross, int n_tiles) {
    bf16* x = buf<bf16>("dx", (size_t)kText * kD);
    bf16* h = buf<bf16>("dh", (size_t)kText * kD);
    bf16* q = buf<bf16>("dq", (size_t)kText * kD);
    bf16* at = buf<bf16>("dat", (size_t)kText * kD);
    bf16* ff = buf<bf16>("dff", (size_t)kText * kFF);
    const bool single = M == 1;
    for (int l = 0; l < cfg.decoder_layers; l++) {
        const Dec& L = dec[l];
        bf16* kl = kc + (size_t)l * kText * kD;
        bf16* vl = vc + (size_t)l * kText * kD;
        tick("other");
        layer_norm(s, x, L.ln1w, L.ln1b, h, M, kD);
        tick("norm");
        Epi e;
        e.bias = L.qb; e.out = q; e.ldo = kD;
        gemm(s, h, kD, L.qw, M, kD, kD, e);
        e = Epi{}; e.out = kl + (size_t)pos0 * kD; e.ldo = kD;
        gemm(s, h, kD, L.kw, M, kD, kD, e);
        e = Epi{}; e.bias = L.vb; e.out = vl + (size_t)pos0 * kD; e.ldo = kD;
        gemm(s, h, kD, L.vw, M, kD, kD, e);
        tick("qkv");
        AttnArgs a{q, kD, kHeadDim, kl, kD, kHeadDim, vl, kD, kHeadDim, at, kD, kHeadDim, 1, self_tiles, single ? 1 : n_tiles,
                   kHeads, 1, pos0, kScale};
        attention(s, a, kHeadDim);
        tick("self_attn");
        e = Epi{}; e.bias = L.ob; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, at, kD, L.ow, M, kD, kD, e);
        tick("o");

        layer_norm(s, x, L.ln2w, L.ln2b, h, M, kD);
        tick("norm");
        e = Epi{}; e.bias = L.xqb; e.out = q; e.ldo = kD;
        gemm(s, h, kD, L.xqw, M, kD, kD, e);
        tick("xq");
        AttnArgs c{q, kD, kHeadDim, xk + (size_t)l * kCtx * kD, kD, kHeadDim, xv + (size_t)l * kCtx * kD, kD, kHeadDim,
                   at, kD, kHeadDim, 1, cross, n_tiles, kHeads, 0, 0, kScale};
        if (single) {
            c.n_tiles = kCrossSplits;
            c.part_o = part_o;
            c.part_ml = part_ml;
            attention(s, c, kHeadDim);
            attention_combine(s, c, 1, kCrossSplits, kHeadDim);
        } else {
            attention(s, c, kHeadDim);
        }
        tick("cross_attn");
        e = Epi{}; e.bias = L.xob; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, at, kD, L.xow, M, kD, kD, e);
        tick("xo");

        layer_norm(s, x, L.ln3w, L.ln3b, h, M, kD);
        tick("norm");
        e = Epi{}; e.bias = L.f1b; e.act = ACT_GELU; e.out = ff; e.ldo = kFF;
        gemm(s, h, kD, L.f1w, M, kFF, kD, e);
        tick("fc1");
        e = Epi{}; e.bias = L.f2b; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, ff, kFF, L.f2w, M, kD, kFF, e);
        tick("fc2");
    }
}

// decoder_rows(1, p, ...) and the next token's pick, with p read from pos_d, so it can replay as a graph.
void CwEngine::step_ops() {
    bf16* x = buf<bf16>("dx", (size_t)kText * kD);
    bf16* h = buf<bf16>("dh", (size_t)kText * kD);
    bf16* q = buf<bf16>("dq", (size_t)kText * kD);
    bf16* at = buf<bf16>("dat", (size_t)kText * kD);
    bf16* ff = buf<bf16>("dff", (size_t)kText * kFF);
    bf16* kn = buf<bf16>("kn", kD);
    bf16* vn = buf<bf16>("vn", kD);
    bf16* logits = buf<bf16>("logits", cfg.vocab);
    const std::string Dn = "model.decoder.";
    bf16* embed = w(Dn + "embed_tokens.weight");
    wh_step_begin(s, pos_d, ids_dev, embed, w(Dn + "embed_positions.weight"), x, step_tile, kD);
    for (int l = 0; l < cfg.decoder_layers; l++) {
        const Dec& L = dec[l];
        layer_norm(s, x, L.ln1w, L.ln1b, h, 1, kD);
        Epi e;
        e.bias = L.qb; e.out = q; e.ldo = kD;
        gemm(s, h, kD, L.qw, 1, kD, kD, e);
        e = Epi{}; e.out = kn; e.ldo = kD;
        gemm(s, h, kD, L.kw, 1, kD, kD, e);
        e = Epi{}; e.bias = L.vb; e.out = vn; e.ldo = kD;
        gemm(s, h, kD, L.vw, 1, kD, kD, e);
        bf16* kl = kc + (size_t)l * kText * kD;
        bf16* vl = vc + (size_t)l * kText * kD;
        wh_store_kv(s, pos_d, kn, vn, kl, vl, kD);
        AttnArgs a{q, kD, kHeadDim, kl, kD, kHeadDim, vl, kD, kHeadDim, at, kD, kHeadDim, 1, step_tile, 1, kHeads, 0, 0, kScale};
        attention(s, a, kHeadDim);
        e = Epi{}; e.bias = L.ob; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, at, kD, L.ow, 1, kD, kD, e);
        layer_norm(s, x, L.ln2w, L.ln2b, h, 1, kD);
        e = Epi{}; e.bias = L.xqb; e.out = q; e.ldo = kD;
        gemm(s, h, kD, L.xqw, 1, kD, kD, e);
        AttnArgs c{q, kD, kHeadDim, xk + (size_t)l * kCtx * kD, kD, kHeadDim, xv + (size_t)l * kCtx * kD, kD, kHeadDim,
                   at, kD, kHeadDim, 1, cross_tiles, kCrossSplits, kHeads, 0, 0, kScale};
        c.part_o = part_o;
        c.part_ml = part_ml;
        attention(s, c, kHeadDim);
        attention_combine(s, c, 1, kCrossSplits, kHeadDim);
        e = Epi{}; e.bias = L.xob; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, at, kD, L.xow, 1, kD, kD, e);
        layer_norm(s, x, L.ln3w, L.ln3b, h, 1, kD);
        e = Epi{}; e.bias = L.f1b; e.act = ACT_GELU; e.out = ff; e.ldo = kFF;
        gemm(s, h, kD, L.f1w, 1, kFF, kD, e);
        e = Epi{}; e.bias = L.f2b; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, ff, kFF, L.f2w, 1, kD, kFF, e);
    }
    layer_norm(s, x, w(Dn + "layer_norm.weight"), w(Dn + "layer_norm.bias"), h, 1, kD);
    Epi e;
    e.bias = suppress; e.out = logits; e.ldo = cfg.vocab;
    gemm(s, h, kD, embed, 1, cfg.vocab, kD, e);
    wh_argmax_step(s, logits, cfg.vocab, ids_dev, pos_d, last_d);
    HIP_OK(hipMemcpyAsync(host_id, last_d, 4, hipMemcpyDeviceToHost, s));
}

// CW_OPBENCH=1: microseconds per launch of each decode op, 200 launches in a row, printed to stderr.
void CwEngine::opbench() {
    bf16* x = buf<bf16>("dx", (size_t)kText * kD);
    bf16* h = buf<bf16>("dh", (size_t)kText * kD);
    bf16* q = buf<bf16>("dq", (size_t)kText * kD);
    bf16* at = buf<bf16>("dat", (size_t)kText * kD);
    bf16* ff = buf<bf16>("dff", (size_t)kText * kFF);
    bf16* logits = buf<bf16>("logits", cfg.vocab);
    const Dec& L = dec[0];
    hipEvent_t a0, a1;
    HIP_OK(hipEventCreate(&a0));
    HIP_OK(hipEventCreate(&a1));
    auto time = [&](const char* tag, auto&& op) {
        for (int i = 0; i < 10; i++) op();
        HIP_OK(hipEventRecord(a0, s));
        for (int i = 0; i < 200; i++) op();
        HIP_OK(hipEventRecord(a1, s));
        HIP_OK(hipEventSynchronize(a1));
        float ms = 0;
        HIP_OK(hipEventElapsedTime(&ms, a0, a1));
        std::fprintf(stderr, "%-12s %8.1f us\n", tag, ms * 1000 / 200);
    };
    int fifty = 50;
    HIP_OK(hipMemcpy(pos_d, &fifty, 4, hipMemcpyHostToDevice));
    time("layer_norm", [&] { layer_norm(s, x, L.ln1w, L.ln1b, h, 1, kD); });
    time("gemv 1280", [&] { Epi e; e.bias = L.qb; e.out = q; e.ldo = kD; gemm(s, h, kD, L.qw, 1, kD, kD, e); });
    time("gemv fc1", [&] { Epi e; e.bias = L.f1b; e.act = ACT_GELU; e.out = ff; e.ldo = kFF; gemm(s, h, kD, L.f1w, 1, kFF, kD, e); });
    time("gemv fc2", [&] { Epi e; e.bias = L.f2b; e.out = at; e.ldo = kD; gemm(s, ff, kFF, L.f2w, 1, kD, kFF, e); });
    time("logits", [&] { Epi e; e.bias = suppress; e.out = logits; e.ldo = cfg.vocab; gemm(s, h, kD, w("model.decoder.embed_tokens.weight"), 1, cfg.vocab, kD, e); });
    time("argmax", [&] { wh_argmax(s, logits, cfg.vocab, last_d); });
    wh_step_begin(s, pos_d, ids_dev, w("model.decoder.embed_tokens.weight"), w("model.decoder.embed_positions.weight"), x, step_tile, kD);
    time("self_attn", [&] {
        AttnArgs a{q, kD, kHeadDim, kc, kD, kHeadDim, vc, kD, kHeadDim, at, kD, kHeadDim, 1, step_tile, 1, kHeads, 0, 0, kScale};
        attention(s, a, kHeadDim);
    });
    time("cross_attn", [&] {
        AttnArgs c{q, kD, kHeadDim, xk, kD, kHeadDim, xv, kD, kHeadDim, at, kD, kHeadDim, 1, cross_tiles, kCrossSplits, kHeads, 0, 0, kScale};
        c.part_o = part_o; c.part_ml = part_ml;
        attention(s, c, kHeadDim);
        attention_combine(s, c, 1, kCrossSplits, kHeadDim);
    });
    (void)hipEventDestroy(a0);
    (void)hipEventDestroy(a1);
}

// A blocking wait on Windows can oversleep a step by a whole scheduler tick (~15 ms, measured: a 0.5 ms
// step took 33 ms), so the token loop spins on the step's event instead.
void CwEngine::wait_step() {
    HIP_OK(hipEventRecord(step_done, s));
    hipError_t e;
    while ((e = hipEventQuery(step_done)) == hipErrorNotReady) YieldProcessor();
    HIP_OK(e);
}

void CwEngine::capture_step() {
    step_ops();  // allocates every buffer the step uses, so none is allocated while capturing
    HIP_OK(hipStreamSynchronize(s));
    hipGraph_t graph;
    HIP_OK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    step_ops();
    HIP_OK(hipStreamEndCapture(s, &graph));
    HIP_OK(hipGraphInstantiate(&step_exec, graph, nullptr, nullptr, 0));
    (void)hipGraphDestroy(graph);
}

std::vector<int> CwEngine::transcribe(const float* pcm, int n, const int* prompt, int prompt_len) {
    const auto t0 = Clock::now();
    if (n < 0 || n > kSamples) throw std::runtime_error("audio longer than 30 s");
    if (prompt_len < 1 || prompt_len >= kText - 1) throw std::runtime_error("prompt length out of range");
    for (int i = 0; i < prompt_len; i++)
        if (prompt[i] < 0 || prompt[i] >= cfg.vocab) throw std::runtime_error("prompt token out of range");
    float* pcm_d = buf<float>("pcm", kSamples);
    HIP_OK(hipMemsetAsync(pcm_d, 0, (size_t)kSamples * 4, s));
    if (n) HIP_OK(hipMemcpyAsync(pcm_d, pcm, (size_t)n * 4, hipMemcpyHostToDevice, s));
    encoder();
    HIP_OK(hipStreamSynchronize(s));
    const double encoder_ms = ms_since(t0);

    const auto t_dec = Clock::now();
    const std::string Dn = "model.decoder.";
    bf16* embed = w(Dn + "embed_tokens.weight");
    bf16* pos = w(Dn + "embed_positions.weight");
    bf16* x = buf<bf16>("dx", (size_t)kText * kD);
    bf16* h = buf<bf16>("dh", (size_t)kText * kD);
    bf16* logits = buf<bf16>("logits", cfg.vocab);
    HIP_OK(hipMemcpyAsync(ids_dev, prompt, (size_t)prompt_len * 4, hipMemcpyHostToDevice, s));
    std::vector<AttnTile> pt;
    for (int r = 0; r < prompt_len; r += 16) pt.push_back({r, std::min(r + 16, prompt_len), 0, INT_MAX});
    std::vector<AttnTile> xt;
    for (int r = 0; r < prompt_len; r += 16) xt.push_back({r, std::min(r + 16, prompt_len), 0, kCtx});
    AttnTile* tiles = buf<AttnTile>("ptiles", pt.size() + xt.size());
    HIP_OK(hipMemcpyAsync(tiles, pt.data(), pt.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(tiles + pt.size(), xt.data(), xt.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    wh_embed_pos(s, ids_dev, embed, pos, 0, x, prompt_len, kD);
    if (prompt_len == 1) decoder_rows(1, 0, row_tile, cross_tiles, 1);
    else decoder_rows(prompt_len, 0, tiles, tiles + pt.size(), (int)pt.size());

    const Epi head = [&] {
        Epi e;
        e.bias = suppress; e.out = logits; e.ldo = cfg.vocab;
        return e;
    }();
    auto next_token = [&](const bf16* row, int slot) {
        layer_norm(s, row, w(Dn + "layer_norm.weight"), w(Dn + "layer_norm.bias"), h, 1, kD);
        gemm(s, h, kD, embed, 1, cfg.vocab, kD, head);
        tick("logits");
        wh_argmax(s, logits, cfg.vocab, ids_dev + slot);
        tick("argmax");
        HIP_OK(hipMemcpyAsync(host_id, ids_dev + slot, 4, hipMemcpyDeviceToHost, s));
        HIP_OK(hipStreamSynchronize(s));
        return *host_id;
    };
    // The same budget the Transformers adapter gives generate(): every position the decoder has, less one.
    const int max_new = kText - prompt_len - 1;
    std::vector<int> out;
    int p = prompt_len;
    int t = next_token(x + (size_t)(prompt_len - 1) * kD, p);
    HIP_OK(hipMemcpyAsync(pos_d, &prompt_len, 4, hipMemcpyHostToDevice, s));
    while (t != cfg.eos && (int)out.size() < max_new) {
        out.push_back(t);
        if ((int)out.size() == max_new) break;
        if (profile) {
            wh_embed_pos(s, ids_dev + p, embed, pos, p, x, 1, kD);
            decoder_rows(1, p, row_tile, cross_tiles, 1);
            p++;
            t = next_token(x, p);
            continue;
        }
        HIP_OK(hipGraphLaunch(step_exec, s));
        wait_step();
        t = *host_id;
    }
    std::string ops;
    for (auto& q : prof) ops += ",\"" + q.first + "_ms\":" + std::to_string(q.second);
    prof.clear();
    char msg[200];
    std::snprintf(msg, sizeof msg, "{\"encoder_ms\":%.2f,\"decoder_ms\":%.2f,\"total_ms\":%.2f,\"prompt\":%d,\"tokens\":%zu}",
                  encoder_ms, ms_since(t_dec), ms_since(t0), prompt_len, out.size());
    stats = msg;
    if (!ops.empty()) stats = stats.substr(0, stats.size() - 1) + ops + "}";
    return out;
}

extern "C" {

CW_API CwEngine* cw_load(const char* path, const char* const* names, const uint64_t* offsets, int count,
                         const CwConfig* config, const int* suppress, int n_suppress, char* err, int err_len) {
    auto* e = new CwEngine();
    try {
        e->cfg = *config;
        e->load(path, names, offsets, count, suppress, n_suppress);
        return e;
    } catch (const std::exception& x) {
        if (err && err_len > 0) std::snprintf(err, err_len, "%s", x.what());
        cw_free(e);
        return nullptr;
    }
}

CW_API int cw_transcribe(CwEngine* e, const float* pcm, int samples, const int* prompt, int prompt_len, int* ids,
                         int max_ids, char* err, int err_len) {
    try {
        const std::vector<int> out = e->transcribe(pcm, samples, prompt, prompt_len);
        if ((int)out.size() > max_ids) throw std::runtime_error("id buffer too small");
        std::memcpy(ids, out.data(), out.size() * 4);
        return (int)out.size();
    } catch (const std::exception& x) {
        if (err && err_len > 0) std::snprintf(err, err_len, "%s", x.what());
        return -1;
    }
}

CW_API const char* cw_stats(CwEngine* e) { return e->stats.c_str(); }

CW_API void cw_free(CwEngine* e) {
    if (!e) return;
    (void)hipDeviceSynchronize();
    for (auto& kv : e->scratch) (void)hipFree(kv.second.first);
    void* devs[] = {e->dev, e->conv1, e->conv2, e->suppress, e->tables, e->filters, e->gmax, e->ids_dev, e->enc_tiles,
                    e->row_tile, e->cross_tiles, e->part_o, e->part_ml, e->xk, e->xv, e->kc, e->vc};
    for (void* p : devs) if (p) (void)hipFree(p);
    if (e->step_exec) (void)hipGraphExecDestroy(e->step_exec);
    if (e->step_done) (void)hipEventDestroy(e->step_done);
    for (void* p : {(void*)e->pos_d, (void*)e->last_d, (void*)e->step_tile}) if (p) (void)hipFree(p);
    if (e->host_id) (void)hipHostFree(e->host_id);
    if (e->s) (void)hipStreamDestroy(e->s);
    delete e;
}

}  // extern "C"
