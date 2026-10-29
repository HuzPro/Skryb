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
constexpr int kMaxRows = 8;       // decode rows per pass: a step, or a verify of [last, drafts...]
constexpr int kCrossSplit = 128;  // keys per split in decode cross-attention
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
    int *ids_round = nullptr, *ids_ver = nullptr;  // [last token, drafts...] and the verified picks per row
    int* host_ids = nullptr;                        // pinned: drafts then verified picks
    hipEvent_t done = nullptr;
    AttnTile *enc_tiles = nullptr, *self_tiles = nullptr, *cross_tiles = nullptr;  // self/cross: per row count 1..kMaxRows
    float *part_o = nullptr, *part_ml = nullptr;
    bf16 *xk = nullptr, *xv = nullptr, *kc = nullptr, *vc = nullptr;  // per decoder layer: cross K/V, self K/V cache
    struct Enc { bf16 *ln1w, *ln1b, *qw, *qb, *kw, *vw, *vb, *ow, *ob, *ln2w, *ln2b, *f1w, *f1b, *f2w, *f2b; };
    struct Dec {
        bf16 *ln1w, *ln1b, *qw, *qb, *kw, *vw, *vb, *ow, *ob;
        bf16 *ln2w, *ln2b, *xqw, *xqb, *xkw, *xvw, *xvb, *xow, *xob;
        bf16 *ln3w, *ln3b, *f1w, *f1b, *f2w, *f2b;
        Q4 q_q, q_k, q_v, q_o, q_xq, q_xo, q_f1, q_f2;  // the 4-bit draft copy
    };
    std::vector<Enc> enc;
    std::vector<Dec> dec;
    Q4 q_head{};
    void* q4_mem = nullptr;
    // Speculative tokens per round, 0 = plain greedy steps. Each round guesses this many with the 4-bit
    // decoder, then one bf16 pass over [last, drafts...] keeps the ones the bf16 model picks too.
    int draft_len = 7;
    int rounds = 0, accepted = 0;
    std::unordered_map<std::string, std::pair<void*, size_t>> scratch;
    std::string stats;
    // CW_PROFILE=1: sync after each op and total the host time per op, reported in stats.
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
    void prepare_draft();
    void encoder();
    void decoder_rows(int M, int pos0, const AttnTile* prompt_tiles, int n_tiles, bool draft);
    void pick(const bf16* x, int M, int* ids, bool draft);
    void wait();
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
        Dec d{};
        d.ln1w = w(p + "self_attn_layer_norm.weight"); d.ln1b = w(p + "self_attn_layer_norm.bias");
        d.qw = w(p + "self_attn.q_proj.weight"); d.qb = w(p + "self_attn.q_proj.bias");
        d.kw = w(p + "self_attn.k_proj.weight");
        d.vw = w(p + "self_attn.v_proj.weight"); d.vb = w(p + "self_attn.v_proj.bias");
        d.ow = w(p + "self_attn.out_proj.weight"); d.ob = w(p + "self_attn.out_proj.bias");
        d.ln2w = w(p + "encoder_attn_layer_norm.weight"); d.ln2b = w(p + "encoder_attn_layer_norm.bias");
        d.xqw = w(p + "encoder_attn.q_proj.weight"); d.xqb = w(p + "encoder_attn.q_proj.bias");
        d.xkw = w(p + "encoder_attn.k_proj.weight");
        d.xvw = w(p + "encoder_attn.v_proj.weight"); d.xvb = w(p + "encoder_attn.v_proj.bias");
        d.xow = w(p + "encoder_attn.out_proj.weight"); d.xob = w(p + "encoder_attn.out_proj.bias");
        d.ln3w = w(p + "final_layer_norm.weight"); d.ln3b = w(p + "final_layer_norm.bias");
        d.f1w = w(p + "fc1.weight"); d.f1b = w(p + "fc1.bias");
        d.f2w = w(p + "fc2.weight"); d.f2b = w(p + "fc2.bias");
        dec.push_back(d);
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
    HIP_OK(hipMalloc(&ids_round, kText * 4));
    HIP_OK(hipMalloc(&ids_ver, kMaxRows * 4));
    HIP_OK(hipHostMalloc(&host_ids, 2 * kMaxRows * 4, hipHostMallocDefault));
    HIP_OK(hipEventCreateWithFlags(&done, hipEventDisableTiming));

    // Fixed attention tiles: encoder query blocks of 16 over every key; for M = 1..kMaxRows decode rows, their
    // causal keys, and the encoder output in splits (merged by attention_combine, the same way for any M).
    std::vector<AttnTile> et, st, ct;
    for (int q = 0; q < kCtx; q += 16) et.push_back({q, std::min(q + 16, kCtx), 0, kCtx});
    for (int m = 1; m <= kMaxRows; m++) {
        st.push_back({0, m, 0, INT_MAX});
        for (int sp = 0; sp < kCrossSplits; sp++) ct.push_back({0, m, sp * kCrossSplit, std::min(kCtx, (sp + 1) * kCrossSplit)});
    }
    HIP_OK(hipMalloc(&enc_tiles, et.size() * sizeof(AttnTile)));
    HIP_OK(hipMalloc(&self_tiles, st.size() * sizeof(AttnTile)));
    HIP_OK(hipMalloc(&cross_tiles, ct.size() * sizeof(AttnTile)));
    HIP_OK(hipMemcpyAsync(enc_tiles, et.data(), et.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(self_tiles, st.data(), st.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(cross_tiles, ct.data(), ct.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    HIP_OK(hipMalloc(&part_o, (size_t)kCrossSplits * kHeads * 16 * kHeadDim * 4));
    HIP_OK(hipMalloc(&part_ml, (size_t)kCrossSplits * kHeads * 16 * 2 * 4));

    const size_t layers = cfg.decoder_layers;
    HIP_OK(hipMalloc(&xk, layers * kCtx * kD * 2));
    HIP_OK(hipMalloc(&xv, layers * kCtx * kD * 2));
    HIP_OK(hipMalloc(&kc, layers * kText * kD * 2));
    HIP_OK(hipMalloc(&vc, layers * kText * kD * 2));
    const auto t_draft = Clock::now();
    if (const char* v = std::getenv("CW_DRAFT")) draft_len = std::max(0, std::min(kMaxRows - 1, std::atoi(v)));
    if (draft_len) prepare_draft();
    HIP_OK(hipStreamSynchronize(s));
    char msg[200];
    std::snprintf(msg, sizeof msg, "{\"weights_ms\":%.1f,\"draft_prep_ms\":%.1f,\"load_ms\":%.1f,\"weights_gib\":%.2f}",
                  weights_ms, ms_since(t_draft), ms_since(t0), bytes / 1073741824.0);
    stats = msg;
}

// The decoder's 4-bit copy (groups of 32 along K) and the logits head's, for drafting.
void CwEngine::prepare_draft() {
    struct Mat { Q4* dst; const bf16* src; int N, K; };
    std::vector<Mat> mats;
    for (Dec& d : dec) {
        mats.push_back({&d.q_q, d.qw, kD, kD});
        mats.push_back({&d.q_k, d.kw, kD, kD});
        mats.push_back({&d.q_v, d.vw, kD, kD});
        mats.push_back({&d.q_o, d.ow, kD, kD});
        mats.push_back({&d.q_xq, d.xqw, kD, kD});
        mats.push_back({&d.q_xo, d.xow, kD, kD});
        mats.push_back({&d.q_f1, d.f1w, kFF, kD});
        mats.push_back({&d.q_f2, d.f2w, kD, kFF});
    }
    mats.push_back({&q_head, w("model.decoder.embed_tokens.weight"), cfg.vocab, kD});
    size_t bytes = 0;
    for (auto& m : mats) bytes += (size_t)m.N * m.K / 2 + ((size_t)m.N * (m.K / 32) * 2 + 255) / 256 * 256;
    HIP_OK(hipMalloc(&q4_mem, bytes));
    char* at = (char*)q4_mem;
    for (auto& m : mats) {
        uint4* q = (uint4*)at;
        at += (size_t)m.N * m.K / 2;
        _Float16* sc = (_Float16*)at;
        at += ((size_t)m.N * (m.K / 32) * 2 + 255) / 256 * 256;
        quantize_q4(s, m.src, m.N, m.K, q, sc);
        *m.dst = Q4{q, sc};
    }
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
    tick("e.stem");

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
        tick("e.qkv");
        AttnArgs a{q, kD, kHeadDim, k, kD, kHeadDim, v, kD, kHeadDim, at, kD, kHeadDim, 1, enc_tiles, n_tiles, kHeads, 0, 0, kScale};
        attention(s, a, kHeadDim);
        tick("e.attn");
        e = Epi{}; e.bias = L.ob; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, at, kD, L.ow, kCtx, kD, kD, e);
        layer_norm(s, x, L.ln2w, L.ln2b, h, kCtx, kD);
        tick("e.o_norm");
        e = Epi{}; e.bias = L.f1b; e.act = ACT_GELU; e.out = ff; e.ldo = kFF;
        gemm(s, h, kD, L.f1w, kCtx, kFF, kD, e);
        e = Epi{}; e.bias = L.f2b; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, ff, kFF, L.f2w, kCtx, kD, kFF, e);
        tick("e.ffn");
    }
    layer_norm(s, x, w(E + "layer_norm.weight"), w(E + "layer_norm.bias"), h, kCtx, kD);
    for (int l = 0; l < cfg.decoder_layers; l++) {
        e = Epi{}; e.out = xk + (size_t)l * kCtx * kD; e.ldo = kD;
        gemm(s, h, kD, dec[l].xkw, kCtx, kD, kD, e);
        e = Epi{}; e.bias = dec[l].xvb; e.out = xv + (size_t)l * kCtx * kD; e.ldo = kD;
        gemm(s, h, kD, dec[l].xvw, kCtx, kD, kD, e);
    }
    tick("e.cross_kv");
}

// M decoder rows at positions pos0.. through every layer, in place in "dx"; their keys and values join the
// cache. Up to kMaxRows rows use the fixed tiles, so a row comes out the same whether it is decoded alone or
// with others (masked keys add exactly nothing); the prompt passes its own tiles. draft: one row, 4-bit weights.
void CwEngine::decoder_rows(int M, int pos0, const AttnTile* prompt_tiles, int n_tiles, bool draft) {
    bf16* x = buf<bf16>("dx", (size_t)kText * kD);
    bf16* h = buf<bf16>("dh", (size_t)kText * kD);
    bf16* q = buf<bf16>("dq", (size_t)kText * kD);
    bf16* at = buf<bf16>("dat", (size_t)kText * kD);
    bf16* ff = buf<bf16>("dff", (size_t)kText * kFF);
    const bool rows = M <= kMaxRows;
    auto mm = [&](const bf16* a, int lda, const bf16* wt, const Q4& wq, int N, int K, const Epi& e) {
        if (draft) gemv_q4(s, a, wq, N, K, e);
        else gemm(s, a, lda, wt, M, N, K, e);
    };
    for (int l = 0; l < cfg.decoder_layers; l++) {
        const Dec& L = dec[l];
        bf16* kl = kc + (size_t)l * kText * kD;
        bf16* vl = vc + (size_t)l * kText * kD;
        layer_norm(s, x, L.ln1w, L.ln1b, h, M, kD);
        Epi e;
        e.bias = L.qb; e.out = q; e.ldo = kD;
        mm(h, kD, L.qw, L.q_q, kD, kD, e);
        e = Epi{}; e.out = kl + (size_t)pos0 * kD; e.ldo = kD;
        mm(h, kD, L.kw, L.q_k, kD, kD, e);
        e = Epi{}; e.bias = L.vb; e.out = vl + (size_t)pos0 * kD; e.ldo = kD;
        mm(h, kD, L.vw, L.q_v, kD, kD, e);
        tick("d.qkv");
        AttnArgs a{q, kD, kHeadDim, kl, kD, kHeadDim, vl, kD, kHeadDim, at, kD, kHeadDim, 1,
                   rows ? self_tiles + (M - 1) : prompt_tiles, rows ? 1 : n_tiles, kHeads, 1, pos0, kScale};
        attention(s, a, kHeadDim);
        tick("d.self_attn");
        e = Epi{}; e.bias = L.ob; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        mm(at, kD, L.ow, L.q_o, kD, kD, e);
        layer_norm(s, x, L.ln2w, L.ln2b, h, M, kD);
        e = Epi{}; e.bias = L.xqb; e.out = q; e.ldo = kD;
        mm(h, kD, L.xqw, L.q_xq, kD, kD, e);
        tick("d.o_xq");
        AttnArgs c{q, kD, kHeadDim, xk + (size_t)l * kCtx * kD, kD, kHeadDim, xv + (size_t)l * kCtx * kD, kD, kHeadDim,
                   at, kD, kHeadDim, 1, prompt_tiles + n_tiles, n_tiles, kHeads, 0, 0, kScale};
        if (rows) {
            c.tiles = cross_tiles + (size_t)(M - 1) * kCrossSplits;
            c.n_tiles = kCrossSplits;
            c.part_o = part_o;
            c.part_ml = part_ml;
            attention(s, c, kHeadDim);
            attention_combine(s, c, M, kCrossSplits, kHeadDim);
        } else {
            attention(s, c, kHeadDim);
        }
        tick("d.cross_attn");
        e = Epi{}; e.bias = L.xob; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        mm(at, kD, L.xow, L.q_xo, kD, kD, e);
        layer_norm(s, x, L.ln3w, L.ln3b, h, M, kD);
        e = Epi{}; e.bias = L.f1b; e.act = ACT_GELU; e.out = ff; e.ldo = kFF;
        mm(h, kD, L.f1w, L.q_f1, kFF, kD, e);
        e = Epi{}; e.bias = L.f2b; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        mm(ff, kFF, L.f2w, L.q_f2, kD, kFF, e);
        tick("d.ffn");
    }
}

// Final norm and the tied head over M rows of x, suppressed tokens masked; each row's argmax lands in ids.
void CwEngine::pick(const bf16* x, int M, int* ids, bool draft) {
    const std::string Dn = "model.decoder.";
    bf16* h = buf<bf16>("ph", (size_t)kMaxRows * kD);
    bf16* logits = buf<bf16>("logits", (size_t)kMaxRows * cfg.vocab);
    layer_norm(s, x, w(Dn + "layer_norm.weight"), w(Dn + "layer_norm.bias"), h, M, kD);
    Epi e;
    e.bias = suppress; e.out = logits; e.ldo = cfg.vocab;
    if (draft) gemv_q4(s, h, q_head, cfg.vocab, kD, e);
    else gemm(s, h, kD, w(Dn + "embed_tokens.weight"), M, cfg.vocab, kD, e);
    wh_argmax(s, logits, cfg.vocab, cfg.vocab, M, ids);
    tick(draft ? "d.draft_pick" : "d.pick");
}

// A blocking wait on Windows can oversleep a short decode round by a scheduler tick, so this spins.
void CwEngine::wait() {
    HIP_OK(hipEventRecord(done, s));
    hipError_t e;
    while ((e = hipEventQuery(done)) == hipErrorNotReady) YieldProcessor();
    HIP_OK(e);
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
    int* ids_prompt = buf<int>("ids_prompt", kText);
    HIP_OK(hipMemcpyAsync(ids_prompt, prompt, (size_t)prompt_len * 4, hipMemcpyHostToDevice, s));
    // The prompt in one pass, with its own causal and cross tiles; its last row picks the first token.
    std::vector<AttnTile> pt;
    for (int r = 0; r < prompt_len; r += 16) pt.push_back({r, std::min(r + 16, prompt_len), 0, INT_MAX});
    const int n_pt = (int)pt.size();
    for (int r = 0; r < prompt_len; r += 16) pt.push_back({r, std::min(r + 16, prompt_len), 0, kCtx});
    AttnTile* tiles = buf<AttnTile>("ptiles", pt.size());
    HIP_OK(hipMemcpyAsync(tiles, pt.data(), pt.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    wh_embed_pos(s, ids_prompt, embed, pos, 0, x, prompt_len, kD);
    decoder_rows(prompt_len, 0, tiles, n_pt, false);
    pick(x + (size_t)(prompt_len - 1) * kD, 1, ids_round, false);
    HIP_OK(hipMemcpyAsync(host_ids, ids_round, 4, hipMemcpyDeviceToHost, s));
    wait();

    // Greedy decoding. With a draft, each round guesses up to draft_len tokens with the 4-bit decoder, then
    // one bf16 pass over [last, drafts...] picks the next token after every row, computed exactly as a single
    // step computes it; drafts up to the first disagreement are kept, plus the bf16 pick after them. The
    // output is the bf16 model's own greedy output either way: the draft only changes how fast it arrives.
    // The budget is the one the Transformers adapter gives generate(): every position, less one.
    const int max_new = kText - prompt_len - 1;
    std::vector<int> out;
    rounds = accepted = 0;
    int t = host_ids[0], p = prompt_len;  // t is the newest token; it sits at position p
    while (t != cfg.eos && (int)out.size() < max_new) {
        out.push_back(t);
        const int k = std::min(draft_len, max_new - (int)out.size());
        if (k <= 0) {
            if ((int)out.size() == max_new) break;
            wh_embed_pos(s, ids_round, embed, pos, p, x, 1, kD);
            decoder_rows(1, p, nullptr, 0, false);
            pick(x, 1, ids_round, false);
            HIP_OK(hipMemcpyAsync(host_ids, ids_round, 4, hipMemcpyDeviceToHost, s));
            wait();
            t = host_ids[0];
            p++;
            continue;
        }
        for (int i = 0; i < k; i++) {
            wh_embed_pos(s, ids_round + i, embed, pos, p + i, x, 1, kD);
            decoder_rows(1, p + i, nullptr, 0, true);
            pick(x, 1, ids_round + i + 1, true);
        }
        wh_embed_pos(s, ids_round, embed, pos, p, x, k + 1, kD);
        decoder_rows(k + 1, p, nullptr, 0, false);
        pick(x, k + 1, ids_ver, false);
        HIP_OK(hipMemcpyAsync(host_ids, ids_round + 1, (size_t)k * 4, hipMemcpyDeviceToHost, s));
        HIP_OK(hipMemcpyAsync(host_ids + kMaxRows, ids_ver, (size_t)(k + 1) * 4, hipMemcpyDeviceToHost, s));
        wait();
        int a = 0;
        while (a < k && host_ids[a] == host_ids[kMaxRows + a]) a++;
        rounds++;
        accepted += a;
        // Verified picks 0..a-1 equal the drafts; pick a comes free. Each is the token after the one before it.
        bool stop = false;
        for (int i = 0; i < a && !stop; i++) {
            const int v = host_ids[kMaxRows + i];
            if (v == cfg.eos || (int)out.size() >= max_new) stop = true;
            else out.push_back(v);
        }
        if (stop) { t = cfg.eos; break; }
        t = host_ids[kMaxRows + a];
        p += a + 1;
        HIP_OK(hipMemcpyAsync(ids_round, ids_ver + a, 4, hipMemcpyDeviceToDevice, s));
    }
    std::string ops;
    for (auto& q : prof) ops += ",\"" + q.first + "_ms\":" + std::to_string(q.second);
    prof.clear();
    char msg[256];
    std::snprintf(msg, sizeof msg,
                  "{\"encoder_ms\":%.2f,\"decoder_ms\":%.2f,\"total_ms\":%.2f,\"prompt\":%d,\"tokens\":%zu,\"rounds\":%d,"
                  "\"accepted\":%d}",
                  encoder_ms, ms_since(t_dec), ms_since(t0), prompt_len, out.size(), rounds, accepted);
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
    void* devs[] = {e->dev, e->conv1, e->conv2, e->suppress, e->tables, e->filters, e->gmax, e->ids_round, e->ids_ver,
                    e->enc_tiles, e->self_tiles, e->cross_tiles, e->part_o, e->part_ml, e->xk, e->xv, e->kc, e->vc,
                    e->q4_mem};
    for (void* p : devs) if (p) (void)hipFree(p);
    if (e->host_ids) (void)hipHostFree(e->host_ids);
    if (e->done) (void)hipEventDestroy(e->done);
    if (e->s) (void)hipStreamDestroy(e->s);
    delete e;
}

}  // extern "C"
