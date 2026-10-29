// Host side of the native Parakeet TDT engine: loads parakeet.bin into VRAM and runs audio -> text.
#define NPK_BUILD
#include "npk.h"
#include "pk_kernels.h"

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

constexpr uint64_t kHeader = 65536;
constexpr int kLayers = 24, kD = 1024, kFF = 4096, kPred = 640, kVocabOut = 8198, kBlank = 8192;
constexpr int kDurations[5] = {0, 1, 2, 3, 4};
constexpr int kMaxSymbolsPerFrame = 10;
constexpr int kWindow = 32;  // frames scored per decoder step
constexpr double kPi = 3.14159265358979323846;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Entry { uint64_t off, bytes; uint32_t host; };

uint16_t to_bf16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    return (uint16_t)((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

// torchaudio.functional.melscale_fbanks(257, 0, 8000, 128, 16000, norm="slaney", mel_scale="slaney"), as [128][257].
std::vector<float> mel_filterbank() {
    const double f_sp = 200.0 / 3.0, min_log_hz = 1000.0, min_log_mel = min_log_hz / f_sp, logstep = std::log(6.4) / 27.0;
    auto to_mel = [&](double hz) { return hz >= min_log_hz ? min_log_mel + std::log(hz / min_log_hz) / logstep : hz / f_sp; };
    auto to_hz = [&](double mel) { return mel >= min_log_mel ? min_log_hz * std::exp(logstep * (mel - min_log_mel)) : f_sp * mel; };
    const int nf = 257, nm = 128;
    std::vector<double> fpts(nm + 2);
    const double mmax = to_mel(8000.0);
    for (int i = 0; i < nm + 2; i++) fpts[i] = to_hz(mmax * i / (nm + 1));
    std::vector<float> fb((size_t)nm * nf);
    for (int k = 0; k < nf; k++) {
        const double freq = 8000.0 * k / (nf - 1);
        for (int m = 0; m < nm; m++) {
            const double down = -(fpts[m] - freq) / (fpts[m + 1] - fpts[m]);
            const double up = (fpts[m + 2] - freq) / (fpts[m + 2] - fpts[m + 1]);
            const double tri = std::max(0.0, std::min(down, up));
            fb[(size_t)m * nf + k] = (float)(tri * 2.0 / (fpts[m + 2] - fpts[m]));
        }
    }
    return fb;
}

}  // namespace

struct NpkEngine {
    hipStream_t s{};
    char* dev = nullptr;
    std::unordered_map<std::string, Entry> table;
    std::vector<uint32_t> vocab_off;
    std::vector<uint8_t> vocab;
    std::unordered_map<std::string, std::pair<void*, size_t>> scratch;
    float *window = nullptr, *cos_sin = nullptr, *fb = nullptr;
    // Decoder: LSTM states (2 layers x 640) in two sets; a step reads set `cur` and writes the other, and an
    // emitted token makes the written set current. One captured graph per direction runs a whole step.
    float *H[2] = {nullptr, nullptr}, *C[2] = {nullptr, nullptr}, *pred = nullptr;
    bf16 *x_dec = nullptr, *h0_bf16 = nullptr, *h1_bf16 = nullptr, *z = nullptr, *logits = nullptr;
    int *label_t = nullptr, *td = nullptr;  // device: label, window start, frames; token and duration per row
    int* io = nullptr;                      // pinned: label, window start, frames, then token/duration rows
    hipGraphExec_t step[2] = {nullptr, nullptr};
    const bf16* step_enc = nullptr;         // the encoder buffer the graphs were captured against
    std::string stats;

    bf16* w(const std::string& name) {
        auto it = table.find(name);
        if (it == table.end() || it->second.host) throw std::runtime_error("missing tensor " + name);
        return (bf16*)(dev + (it->second.off - kHeader));
    }
    float* wf(const std::string& name) { return (float*)w(name); }

    template <class T>
    T* buf(const char* name, size_t count) {
        auto& slot = scratch[name];
        size_t bytes = count * sizeof(T);
        if (slot.second < bytes) {
            HIP_OK(hipStreamSynchronize(s));
            if (slot.first) HIP_OK(hipFree(slot.first));
            bytes = std::max(bytes, slot.second * 3 / 2);
            HIP_OK(hipMalloc(&slot.first, bytes));
            slot.second = bytes;
        }
        return (T*)slot.first;
    }

    void load(const char* path);
    std::string transcribe(const float* pcm, int n);

   private:
    void layer(int i, bf16* x, int T, const bf16* pe);
    void capture_steps(const bf16* enc);
};

void NpkEngine::load(const char* path) {
    const auto t0 = Clock::now();
    HIP_OK(hipSetDevice(0));
    HIP_OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<char> head(kHeader);
    DWORD got = 0;
    if (!ReadFile(f, head.data(), (DWORD)kHeader, &got, nullptr) || got != kHeader || std::memcmp(head.data(), "NPK00001", 8))
        throw std::runtime_error("not a parakeet.bin");
    uint32_t count;
    std::memcpy(&count, head.data() + 8, 4);
    uint64_t dev_end = kHeader;
    for (uint32_t i = 0; i < count; i++) {
        const char* p = head.data() + 12 + i * 84;
        Entry e;
        std::memcpy(&e.off, p + 64, 8);
        std::memcpy(&e.bytes, p + 72, 8);
        std::memcpy(&e.host, p + 80, 4);
        table[std::string(p, strnlen(p, 64))] = e;
        if (!e.host) dev_end = std::max(dev_end, e.off + e.bytes);
    }
    // The device image is small (about 1.2 GB): one read into host memory, one copy.
    const uint64_t dev_bytes = dev_end - kHeader;
    std::vector<char> host(dev_bytes);
    LARGE_INTEGER at;
    at.QuadPart = (LONGLONG)kHeader;
    SetFilePointerEx(f, at, nullptr, FILE_BEGIN);
    for (uint64_t done = 0; done < dev_bytes;) {
        const DWORD want = (DWORD)std::min<uint64_t>(dev_bytes - done, 1ull << 30);
        if (!ReadFile(f, host.data() + done, want, &got, nullptr) || got == 0) throw std::runtime_error("short read");
        done += got;
    }
    HIP_OK(hipMalloc(&dev, dev_bytes));
    HIP_OK(hipMemcpy(dev, host.data(), dev_bytes, hipMemcpyHostToDevice));
    auto read_host = [&](const char* name, void* dst) {
        const Entry& e = table.at(name);
        LARGE_INTEGER pos;
        pos.QuadPart = (LONGLONG)e.off;
        SetFilePointerEx(f, pos, nullptr, FILE_BEGIN);
        if (!ReadFile(f, dst, (DWORD)e.bytes, &got, nullptr) || got != e.bytes) throw std::runtime_error("short read");
    };
    vocab_off.resize(table.at("vocab.offsets").bytes / 4);
    read_host("vocab.offsets", vocab_off.data());
    vocab.resize(table.at("vocab.bytes").bytes);
    read_host("vocab.bytes", vocab.data());
    CloseHandle(f);

    // NeMo's fixed featurizer buffers: a symmetric 400-sample Hann window centred in 512, DFT tables, mel bands.
    std::vector<float> win(512, 0.f), cs(1024), mel = mel_filterbank();
    for (int i = 0; i < 400; i++) win[56 + i] = (float)(0.5 - 0.5 * std::cos(2.0 * kPi * i / 399.0));
    for (int k = 0; k < 512; k++) {
        cs[k] = (float)std::cos(2.0 * kPi * k / 512.0);
        cs[512 + k] = (float)std::sin(2.0 * kPi * k / 512.0);
    }
    HIP_OK(hipMalloc(&window, 512 * 4));
    HIP_OK(hipMalloc(&cos_sin, 1024 * 4));
    HIP_OK(hipMalloc(&fb, mel.size() * 4));
    HIP_OK(hipMemcpy(window, win.data(), 512 * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(cos_sin, cs.data(), 1024 * 4, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(fb, mel.data(), mel.size() * 4, hipMemcpyHostToDevice));
    for (float** p : {&H[0], &H[1], &C[0], &C[1]}) HIP_OK(hipMalloc(p, 2 * kPred * 4));
    HIP_OK(hipMalloc(&pred, kPred * 4));
    HIP_OK(hipMalloc(&x_dec, kPred * 2));
    HIP_OK(hipMalloc(&h0_bf16, kPred * 2));
    HIP_OK(hipMalloc(&h1_bf16, kPred * 2));
    HIP_OK(hipMalloc(&z, (size_t)kWindow * kPred * 2));
    HIP_OK(hipMalloc(&logits, (size_t)kWindow * kVocabOut * 2));
    HIP_OK(hipMalloc(&label_t, 3 * 4));
    HIP_OK(hipMalloc(&td, kWindow * 2 * 4));
    HIP_OK(hipHostMalloc(&io, (3 + kWindow * 2) * 4, hipHostMallocDefault));
    char msg[160];
    std::snprintf(msg, sizeof msg, "{\"load_ms\":%.1f,\"device_mb\":%.0f}", ms_since(t0), dev_bytes / 1048576.0);
    stats = msg;
}

// FF1 -> self-attention -> conv -> FF2 -> layer norm, NeMo's order, x updated in place.
void NpkEngine::layer(int i, bf16* x, int T, const bf16* pe) {
    const std::string p = "L" + std::to_string(i) + ".";
    bf16* hn = buf<bf16>("hn", (size_t)T * kD);
    bf16* ff = buf<bf16>("ff", (size_t)T * kFF);
    Epi e;
    auto feed_forward = [&](const char* tag) {
        const std::string q = p + tag;
        layer_norm(s, x, w(q + ".ln.w"), w(q + ".ln.b"), hn, T, kD);
        e = Epi{}; e.act = ACT_SILU; e.out = ff; e.ldo = kFF;
        gemm(s, hn, kD, w(q + ".w1"), T, kFF, kD, e);
        e = Epi{}; e.scale = 0.5f; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
        gemm(s, ff, kFF, w(q + ".w2"), T, kD, kFF, e);
    };
    feed_forward("ff1");

    layer_norm(s, x, w(p + "ln_att.w"), w(p + "ln_att.b"), hn, T, kD);
    bf16* qkv = buf<bf16>("qkv", (size_t)T * 3 * kD);
    e = Epi{}; e.out = qkv; e.ldo = 3 * kD;
    gemm(s, hn, kD, w(p + "qkv"), T, 3 * kD, kD, e);
    bf16* pos = buf<bf16>("pos", (size_t)(2 * T - 1) * kD);
    e = Epi{}; e.out = pos; e.ldo = kD;
    gemm(s, pe, kD, w(p + "pos"), 2 * T - 1, kD, kD, e);
    bf16* att = buf<bf16>("att", (size_t)T * kD);
    pk_relpos_attention(s, qkv, pos, w(p + "u"), w(p + "v"), T, att);
    e = Epi{}; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
    gemm(s, att, kD, w(p + "out"), T, kD, kD, e);

    layer_norm(s, x, w(p + "ln_conv.w"), w(p + "ln_conv.b"), hn, T, kD);
    bf16* g = buf<bf16>("glu_in", (size_t)T * 2 * kD);
    e = Epi{}; e.out = g; e.ldo = 2 * kD;
    gemm(s, hn, kD, w(p + "pw1"), T, 2 * kD, kD, e);
    pk_glu_dwconv_silu(s, g, T, wf(p + "dw.w"), wf(p + "dw.b"), att);
    e = Epi{}; e.res = x; e.res_ld = kD; e.out = x; e.ldo = kD;
    gemm(s, att, kD, w(p + "pw2"), T, kD, kD, e);

    feed_forward("ff2");
    layer_norm(s, x, w(p + "ln_out.w"), w(p + "ln_out.b"), x, T, kD);
}

// A decoder step as one graph per direction: label and window in, the LSTM from set `from` into the other set,
// the joint over kWindow frames, token and duration per frame out to pinned memory.
void NpkEngine::capture_steps(const bf16* enc) {
    for (int from = 0; from < 2; from++) {
        if (step[from]) HIP_OK(hipGraphExecDestroy(step[from]));
        hipGraph_t graph;
        HIP_OK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
        HIP_OK(hipMemcpyAsync(label_t, io, 3 * 4, hipMemcpyHostToDevice, s));
        pk_embed(s, w("dec.embed"), label_t, x_dec);
        float *h = H[from], *c = C[from], *hn = H[1 - from], *cn = C[1 - from];
        pk_lstm_step(s, x_dec, h, c, w("lstm0.ih"), w("lstm0.hh"), wf("lstm0.b"), hn, cn, h0_bf16);
        pk_lstm_step(s, h0_bf16, h + kPred, c + kPred, w("lstm1.ih"), w("lstm1.hh"), wf("lstm1.b"), hn + kPred, cn + kPred,
                     h1_bf16);
        pk_linear640(s, hn + kPred, w("joint.pred.w"), w("joint.pred.b"), pred);
        pk_joint_window(s, enc, pred, label_t + 1, kWindow, z);
        Epi e;
        e.bias = w("joint.out.b"); e.out = logits; e.ldo = kVocabOut;
        gemm(s, z, kPred, w("joint.out.w"), kWindow, kVocabOut, kPred, e);
        pk_token_duration(s, logits, kWindow, td);
        HIP_OK(hipMemcpyAsync(io + 3, td, kWindow * 2 * 4, hipMemcpyDeviceToHost, s));
        HIP_OK(hipStreamEndCapture(s, &graph));
        HIP_OK(hipGraphInstantiate(&step[from], graph, nullptr, nullptr, 0));
        (void)hipGraphDestroy(graph);
    }
    step_enc = enc;
}

std::string NpkEngine::transcribe(const float* pcm, int n) {
    const auto t0 = Clock::now();
    if (n < 2) throw std::runtime_error("audio too short");
    const int frames = 1 + n / 160;
    float* pcm_d = buf<float>("pcm", n);
    HIP_OK(hipMemcpyAsync(pcm_d, pcm, (size_t)n * 4, hipMemcpyHostToDevice, s));
    float* mel = buf<float>("mel", (size_t)frames * 128);
    pk_log_mel(s, pcm_d, n, frames, window, cos_sin, fb, mel);
    bf16* feat = buf<bf16>("feat", (size_t)frames * 128);
    pk_normalize(s, mel, frames, feat);

    const int t1 = (frames - 1) / 2 + 1, t2 = (t1 - 1) / 2 + 1, T = (t2 - 1) / 2 + 1;
    bf16* c0 = buf<bf16>("c0", (size_t)t1 * 64 * 256);
    pk_conv0(s, feat, frames, w("sub.conv0.w"), w("sub.conv0.b"), c0);
    bf16* d1 = buf<bf16>("d1", (size_t)t2 * 32 * 256);
    pk_dwconv(s, c0, t1, 64, w("sub.dw1.w"), w("sub.dw1.b"), d1);
    bf16* p1 = buf<bf16>("p1", (size_t)t2 * 32 * 256);
    Epi e;
    e.bias = w("sub.pw1.b"); e.act = ACT_RELU; e.out = p1; e.ldo = 256;
    gemm(s, d1, 256, w("sub.pw1.w"), t2 * 32, 256, 256, e);
    bf16* d2 = buf<bf16>("d2", (size_t)T * 16 * 256);
    pk_dwconv(s, p1, t2, 32, w("sub.dw2.w"), w("sub.dw2.b"), d2);
    bf16* p2 = buf<bf16>("p2", (size_t)T * 16 * 256);
    e = Epi{}; e.bias = w("sub.pw2.b"); e.act = ACT_RELU; e.out = p2; e.ldo = 256;
    gemm(s, d2, 256, w("sub.pw2.w"), T * 16, 256, 256, e);
    bf16* x = buf<bf16>("x", (size_t)T * kD);
    e = Epi{}; e.bias = w("sub.out.b"); e.out = x; e.ldo = kD;
    gemm(s, p2, 4096, w("sub.out.w"), T, kD, 4096, e);

    // Relative positions t - 1 down to -(t - 1), sin on even and cos on odd dims, in bf16 like the model.
    std::vector<uint16_t> pe_host((size_t)(2 * T - 1) * kD);
    for (int r = 0; r < 2 * T - 1; r++) {
        const float pos = (float)(T - 1 - r);
        for (int i = 0; i < kD / 2; i++) {
            const float div = std::exp((float)(2 * i) * -(std::log(10000.0f) / kD));
            pe_host[(size_t)r * kD + 2 * i] = to_bf16_bits(std::sin(pos * div));
            pe_host[(size_t)r * kD + 2 * i + 1] = to_bf16_bits(std::cos(pos * div));
        }
    }
    bf16* pe = buf<bf16>("pe", pe_host.size());
    HIP_OK(hipMemcpyAsync(pe, pe_host.data(), pe_host.size() * 2, hipMemcpyHostToDevice, s));
    for (int i = 0; i < kLayers; i++) layer(i, x, T, pe);
    bf16* enc = buf<bf16>("enc", (size_t)T * kPred);
    e = Epi{}; e.bias = w("joint.enc.b"); e.out = enc; e.ldo = kPred;
    gemm(s, x, kD, w("joint.enc.w"), T, kPred, kD, e);
    HIP_OK(hipStreamSynchronize(s));
    const double encoder_ms = ms_since(t0);

    // TDT greedy decoding (NeMo's GreedyTDTInfer). The prediction only changes when a token is emitted, so one
    // step scores a window of frames and the walk over blanks happens on the host.
    if (step_enc != enc) capture_steps(enc);
    HIP_OK(hipMemsetAsync(H[0], 0, 2 * kPred * 4, s));
    HIP_OK(hipMemsetAsync(C[0], 0, 2 * kPred * 4, s));
    int cur = 0, label = kBlank, t = 0, base = 0, steps = 0;
    bool stale = true;
    std::vector<int> tokens;
    auto run_step = [&](int from) {
        io[0] = label;
        io[1] = from;
        io[2] = T;
        HIP_OK(hipGraphLaunch(step[cur], s));
        HIP_OK(hipStreamSynchronize(s));
        base = from;
        stale = false;
        steps++;
    };
    while (t < T) {
        const int frame_start = t;
        int symbols = 0;
        bool need_loop = true;
        while (need_loop && symbols < kMaxSymbolsPerFrame && t < T) {
            if (stale || t >= base + kWindow) run_step(t);
            const int k = io[3 + 2 * (t - base)], skip = kDurations[io[3 + 2 * (t - base) + 1]];
            if (k == kBlank) {
                need_loop = false;
            } else {
                tokens.push_back(k);
                cur ^= 1;
                label = k;
                stale = true;
            }
            symbols++;
            t += skip;
            need_loop = need_loop && skip == 0;
        }
        if (t <= frame_start) t = frame_start + 1;
    }

    std::string text;
    for (int id : tokens)
        if (id >= 0 && id + 1 < (int)vocab_off.size())
            text.append((const char*)vocab.data() + vocab_off[id], vocab_off[id + 1] - vocab_off[id]);
    // SentencePiece: U+2581 marks a word start.
    std::string out;
    for (size_t i = 0; i < text.size();) {
        if (i + 2 < text.size() + 0 && (unsigned char)text[i] == 0xE2 && (unsigned char)text[i + 1] == 0x96 &&
            (unsigned char)text[i + 2] == 0x81) {
            out += ' ';
            i += 3;
        } else {
            out += text[i++];
        }
    }
    const size_t first = out.find_first_not_of(' '), last = out.find_last_not_of(' ');
    out = first == std::string::npos ? "" : out.substr(first, last - first + 1);
    char msg[200];
    std::snprintf(msg, sizeof msg, "{\"encoder_ms\":%.2f,\"total_ms\":%.2f,\"frames\":%d,\"tokens\":%zu,\"steps\":%d}",
                  encoder_ms, ms_since(t0), T, tokens.size(), steps);
    stats = msg;
    return out;
}

extern "C" {

NPK_API NpkEngine* npk_load(const char* path, char* err, int err_len) {
    auto* e = new NpkEngine();
    try {
        e->load(path);
        return e;
    } catch (const std::exception& x) {
        if (err && err_len > 0) std::snprintf(err, err_len, "%s", x.what());
        npk_free(e);
        return nullptr;
    }
}

NPK_API int npk_transcribe(NpkEngine* e, const float* pcm, int samples, char* text, int text_len) {
    try {
        const std::string t = e->transcribe(pcm, samples);
        if ((int)t.size() >= text_len) throw std::runtime_error("text buffer too small");
        std::memcpy(text, t.c_str(), t.size() + 1);
        return (int)t.size();
    } catch (const std::exception& x) {
        if (text && text_len > 0) std::snprintf(text, text_len, "%s", x.what());
        return -1;
    }
}

NPK_API const char* npk_stats(NpkEngine* e) { return e->stats.c_str(); }

NPK_API void npk_free(NpkEngine* e) {
    if (!e) return;
    (void)hipDeviceSynchronize();
    for (auto& kv : e->scratch) (void)hipFree(kv.second.first);
    for (auto& g : e->step) if (g) (void)hipGraphExecDestroy(g);
    void* devs[] = {e->dev, e->window, e->cos_sin, e->fb, e->H[0], e->H[1], e->C[0], e->C[1], e->pred, e->x_dec,
                    e->h0_bf16, e->h1_bf16, e->z, e->logits, e->label_t, e->td};
    for (void* p : devs) if (p) (void)hipFree(p);
    if (e->io) (void)hipHostFree(e->io);
    if (e->s) (void)hipStreamDestroy(e->s);
    delete e;
}

}  // extern "C"
