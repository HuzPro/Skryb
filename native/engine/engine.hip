// Host side: loads qasr.bin straight into VRAM and runs audio -> text.
#define QASR_BUILD
#include "qasr.h"
#include "kernels.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "stream_load.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define HIP_OK(x)                                                                              \
    do {                                                                                       \
        hipError_t e_ = (x);                                                                   \
        if (e_ != hipSuccess) throw std::runtime_error(std::string(#x) + ": " + hipGetErrorString(e_)); \
    } while (0)

namespace {

constexpr uint64_t kHeader = 65536;
constexpr int kChunkBatch = 32;  // conv stem chunks per pass: bounds its scratch to ~130 MB
constexpr int kMaxRows = 8;       // decode / verify rows per step (the GEMV path)
constexpr int kSplitKeys = 64;    // keys per split in decode attention
constexpr int kMaxSplits = 512;   // 32768 positions
constexpr int kPrefillRows = 2048;  // prefill runs in row chunks: scratch stays flat for any length
constexpr size_t kKvKeepBytes = 512ull << 20;  // a cache grown past this is freed after the call

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }
int cdiv(int a, int b) { return (a + b - 1) / b; }

struct Entry { uint64_t off, bytes; uint32_t host; };

// The model's shape, from the pack's "config" entry. A pack without one is Qwen3-ASR-1.7B, as every pack
// was before Jais-ASR. Heads are 64 wide in the encoder and 128 in the decoder.
struct Cfg {
    int enc_layers = 24, enc_d = 1024, enc_heads = 16, enc_ffn = 4096, enc_out = 2048;
    int dec_layers = 28, dec_d = 2048, q_heads = 16, kv_heads = 8, dec_ffn = 6144, vocab = 151936;
    int eos1 = 151643, eos2 = 151645;
    int qk_norm = 1;      // per-head RMSNorm on queries and keys before rope
    int norm_groups = 1;  // the decoder's RMSNorms normalise this many slices of a row on their own
    int adapter = 0;      // LayerNorm, Linear, GELU, Linear from enc_out to dec_d after the audio tower
    int q4 = 0;           // decoder matrices, token embeddings and head packed in 4 bits; no draft
    float rope_theta = 1000000.f;
    int qkv_w() const { return (q_heads + 2 * kv_heads) * 128; }
    int q_w() const { return q_heads * 128; }
    int kv_w() const { return kv_heads * 128; }
};
constexpr int kCfgInts = 18;

// Whisper / transformers mel_filter_bank(201 bins, 128 mels, 0-8000 Hz, slaney scale and norm), as [201][128].
std::vector<float> mel_filters() {
    const double logstep = 27.0 / std::log(6.4);
    auto to_mel = [&](double hz) { return hz >= 1000.0 ? 15.0 + std::log(hz / 1000.0) * logstep : 3.0 * hz / 200.0; };
    auto to_hz = [&](double mel) { return mel >= 15.0 ? 1000.0 * std::exp((mel - 15.0) / logstep) : 200.0 * mel / 3.0; };
    const int nf = 201, nm = 128;
    std::vector<double> ff(nm + 2);
    const double lo = to_mel(0.0), hi = to_mel(8000.0);
    for (int i = 0; i < nm + 2; i++) ff[i] = to_hz(lo + (hi - lo) * i / (nm + 1));
    std::vector<float> out(nf * nm);
    for (int k = 0; k < nf; k++) {
        const double fft = 8000.0 * k / (nf - 1);
        for (int j = 0; j < nm; j++) {
            const double down = (fft - ff[j]) / (ff[j + 1] - ff[j]);
            const double up = (ff[j + 2] - fft) / (ff[j + 2] - ff[j + 1]);
            const double w = std::max(0.0, std::min(down, up)) * 2.0 / (ff[j + 2] - ff[j]);
            out[k * nm + j] = (float)w;
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

uint16_t to_bf16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

// UTF-8 bytes to code points, invalid sequences become U+FFFD (what from_utf8_lossy does).
std::u32string utf8_decode(const std::string& s) {
    std::u32string out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        char32_t cp = len == 1 ? c : len == 2 ? (c & 0x1f) : len == 3 ? (c & 0x0f) : (c & 0x07);
        bool ok = len > 0 && i + len <= s.size();
        for (int k = 1; ok && k < len; k++) {
            unsigned char cc = s[i + k];
            if ((cc >> 6) != 2) ok = false;
            else cp = (cp << 6) | (cc & 0x3f);
        }
        if (!ok) { out += char32_t(0xFFFD); i += 1; continue; }
        out += cp;
        i += len;
    }
    return out;
}

std::string utf8_encode(const std::u32string& s) {
    std::string o;
    for (char32_t c : s) {
        if (c < 0x80) o += (char)c;
        else if (c < 0x800) { o += (char)(0xc0 | (c >> 6)); o += (char)(0x80 | (c & 0x3f)); }
        else if (c < 0x10000) { o += (char)(0xe0 | (c >> 12)); o += (char)(0x80 | ((c >> 6) & 0x3f)); o += (char)(0x80 | (c & 0x3f)); }
        else { o += (char)(0xf0 | (c >> 18)); o += (char)(0x80 | ((c >> 12) & 0x3f)); o += (char)(0x80 | ((c >> 6) & 0x3f)); o += (char)(0x80 | (c & 0x3f)); }
    }
    return o;
}

// Qwen3-ASR's own post-processing (qwen_asr/inference/utils.py): collapse runs of more than
// 20 repeats of one character or of one pattern up to 20 characters long.
std::u32string fix_char_repeats(const std::u32string& s, size_t thresh) {
    std::u32string res;
    size_t i = 0, n = s.size();
    while (i < n) {
        size_t count = 1;
        while (i + count < n && s[i + count] == s[i]) count++;
        if (count > thresh) res += s[i];
        else res += s.substr(i, count);
        i += count;
    }
    return res;
}

std::u32string fix_pattern_repeats(const std::u32string& s, size_t thresh, size_t max_len = 20) {
    const size_t n = s.size(), min_chars = thresh * 2;
    if (n < min_chars) return s;
    std::u32string result;
    size_t i = 0;
    bool found = false;
    while (i + min_chars <= n) {
        for (size_t k = 1; k <= max_len; k++) {
            if (i + k * thresh > n) break;
            bool valid = true;
            for (size_t rep = 1; rep < thresh && valid; rep++)
                if (s.compare(i + rep * k, k, s, i, k) != 0) valid = false;
            if (valid) {
                size_t end = i + thresh * k;
                while (end + k <= n && s.compare(end, k, s, i, k) == 0) end += k;
                result += s.substr(i, k);
                result += fix_pattern_repeats(s.substr(end), thresh, max_len);
                i = n;
                found = true;
                break;
            }
        }
        if (found) break;
        result += s[i];
        i++;
    }
    if (!found) result += s.substr(std::min(i, n));
    return result;
}

bool is_space(char32_t c) {
    return c == U' ' || (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x1f) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

std::u32string strip(const std::u32string& s) {
    size_t a = 0, z = s.size();
    while (a < z && is_space(s[a])) a++;
    while (z > a && is_space(s[z - 1])) z--;
    return s.substr(a, z - a);
}

}  // namespace

struct QasrEngine {
    Cfg cfg;
    hipStream_t s{};
    char* dev = nullptr;
    std::unordered_map<std::string, Entry> table;
    std::vector<uint32_t> vocab_off;
    std::vector<uint8_t> vocab;
    std::vector<int> prefix, suffix;
    std::unordered_map<std::string, std::pair<void*, size_t>> scratch;
    float *tables = nullptr, *filters = nullptr, *inv_freq = nullptr;
    bf16* pos13 = nullptr;
    int* conv3_map = nullptr;
    unsigned* gmax = nullptr;
    unsigned long long* argkeys = nullptr;
    int* ids_one = nullptr;
    int* host_tok = nullptr;
    std::string stats;
    int rounds = 0, accepted = 0;
    // QASR_PROFILE=1: sync after each op and total host time per op, reported in stats.
    bool profile = std::getenv("QASR_PROFILE") != nullptr;
    std::vector<std::pair<std::string, double>> prof;
    Clock::time_point prof_t = Clock::now();
    void tick(const char* tag) {
        if (!profile) return;
        HIP_OK(hipStreamSynchronize(s));
        const double ms = ms_since(prof_t);
        prof_t = Clock::now();
        for (auto& p : prof) if (p.first == tag) { p.second += ms; return; }
        prof.push_back({tag, ms});
    }  // last call's speculative rounds and drafts kept
    size_t kv_pos = 0;  // positions the KV cache holds per layer
    // Speculative tokens per round, 0 = plain greedy steps. 7 measured fastest: each 4-bit draft
    // step costs ~3.7 ms, so longer rounds lose more to rejected tails than one verify pass saves.
    int draft_len = 7;
    AttnTile* dec_tiles = nullptr;  // [rows - 1][split]
    float *part_o = nullptr, *part_ml = nullptr;
    int* ids_round = nullptr;  // [last token, drafts...]
    int* ids_ver = nullptr;
    // The model's own odds of "?" over "." where the transcript ends: the logits of both tokens on
    // every row that picks a token, kept for the last one when it is a full stop or question mark.
    int dot_id = -1, question_id = -1;
    float* pair = nullptr;           // device [kMaxRows][2]
    float question_probability = -1.f;  // -1 when the transcript does not end in "." or "?"
    struct Layer { bf16 *ln1, *qkv, *qn, *kn, *o, *ln2, *gu, *down; Q4 q_qkv, q_o, q_gu, q_down; };
    std::vector<Layer> layers;
    Q4 q_head{}, q_embed{};
    void* q4_mem = nullptr;

    bf16* w(const std::string& name) {
        auto it = table.find(name);
        if (it == table.end() || it->second.host) throw std::runtime_error("missing tensor " + name);
        return (bf16*)(dev + (it->second.off - kHeader));
    }

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
    std::string transcribe(const float* pcm, int n) { return transcribe(pcm, n, prefix, suffix); }
    std::string transcribe(const float* pcm, int n, const std::vector<int>& pre, const std::vector<int>& suf);

   private:
    void read_host(HANDLE f, const Entry& e, void* dst);
    void read_config(HANDLE f);
    Q4 packed_q4(const std::string& name);
    void embed_rows(const int* ids, const bf16* audio, int audio_start, bf16* out, int rows);
    void encoder(int T, const AttnTile* tiles, int n_tiles);
    void prepare_decoder();
    void decoder_layers(int M, int pos0, const AttnTile* tiles, int n_tiles, bool draft = false);
    void logits_argmax(const bf16* x, int M, int* ids, bool draft = false);
    std::vector<int> generate(const bf16* last_row, int L, int max_new);
    std::string detokenize(const std::vector<int>& ids);
};

void QasrEngine::read_host(HANDLE f, const Entry& e, void* dst) {
    LARGE_INTEGER at;
    at.QuadPart = (LONGLONG)e.off;
    SetFilePointerEx(f, at, nullptr, FILE_BEGIN);
    DWORD got = 0;
    if (!ReadFile(f, dst, (DWORD)e.bytes, &got, nullptr) || got != e.bytes) throw std::runtime_error("short read");
}

void QasrEngine::read_config(HANDLE f) {
    auto it = table.find("config");
    if (it == table.end()) return;
    if (it->second.bytes != kCfgInts * 4) throw std::runtime_error("config entry of the wrong size");
    int32_t c[kCfgInts];
    read_host(f, it->second, c);
    cfg.enc_layers = c[0]; cfg.enc_d = c[1]; cfg.enc_heads = c[2]; cfg.enc_ffn = c[3]; cfg.enc_out = c[4];
    cfg.dec_layers = c[5]; cfg.dec_d = c[6]; cfg.q_heads = c[7]; cfg.kv_heads = c[8]; cfg.dec_ffn = c[9];
    cfg.vocab = c[10]; cfg.eos1 = c[11]; cfg.eos2 = c[12]; cfg.qk_norm = c[13]; cfg.norm_groups = c[14];
    cfg.adapter = c[15]; cfg.q4 = c[16];
    std::memcpy(&cfg.rope_theta, &c[17], 4);
}

Q4 QasrEngine::packed_q4(const std::string& name) {
    return Q4{(const uint4*)w(name + ".q"), (const f16*)w(name + ".s")};
}

void QasrEngine::embed_rows(const int* ids, const bf16* audio, int audio_start, bf16* out, int rows) {
    if (cfg.q4) embed_q4(s, ids, q_embed, audio, audio_start, out, rows, cfg.dec_d);
    else embed(s, ids, w("dec.embed"), audio, audio_start, out, rows, cfg.dec_d);
}

void QasrEngine::load(const char* path) {
    const auto t0 = Clock::now();
    HIP_OK(hipSetDevice(0));
    HIP_OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    const double hip_ms = ms_since(t0);
    if (const std::string bad = gpu_self_check(s); !bad.empty()) throw std::runtime_error("GPU self-check failed: " + bad);

    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<char> head(kHeader);
    DWORD got = 0;
    if (!ReadFile(f, head.data(), (DWORD)kHeader, &got, nullptr) || got != kHeader || std::memcmp(head.data(), "QASR0001", 8))
        throw std::runtime_error("not a qasr.bin");
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
    const uint64_t dev_bytes = dev_end - kHeader;
    const auto t_alloc = Clock::now();
    HIP_OK(hipMalloc(&dev, dev_bytes));
    const double alloc_ms = ms_since(t_alloc);

    const auto t_read = Clock::now();
    const char* io = std::getenv("QASR_IO");
    const char* readers = std::getenv("QASR_READERS");
    stream_to_device(path, kHeader, io && std::string(io) == "unbuffered", readers ? std::max(1, std::atoi(readers)) : 4, dev,
                     dev_bytes);

    auto host_entry = [&](const char* name) -> const Entry& { return table.at(name); };
    vocab_off.resize(host_entry("vocab.offsets").bytes / 4);
    read_host(f, host_entry("vocab.offsets"), vocab_off.data());
    vocab.resize(host_entry("vocab.bytes").bytes);
    read_host(f, host_entry("vocab.bytes"), vocab.data());
    for (int id = 0; id + 1 < (int)vocab_off.size(); id++) {
        if (vocab_off[id + 1] - vocab_off[id] != 1) continue;
        const char c = (char)vocab[vocab_off[id]];
        if (c == '.' && dot_id < 0) dot_id = id;
        if (c == '?' && question_id < 0) question_id = id;
    }
    // The head's odds of "?" over "." need its bf16 rows.
    read_config(f);
    if (cfg.q4) dot_id = question_id = -1;
    prefix.resize(host_entry("prompt.prefix").bytes / 4);
    read_host(f, host_entry("prompt.prefix"), prefix.data());
    suffix.resize(host_entry("prompt.suffix").bytes / 4);
    read_host(f, host_entry("prompt.suffix"), suffix.data());
    CloseHandle(f);

    // Constants computed on the host, the same way the reference computes them.
    const auto tab = dft_tables();
    const auto fil = mel_filters();
    std::vector<float> inv(64);
    for (int i = 0; i < 64; i++) inv[i] = 1.0f / std::pow(cfg.rope_theta, (float)(2 * i) / 128.0f);
    const int ed = cfg.enc_d, half = ed / 2;
    std::vector<uint16_t> pos(13 * ed);
    const float inc = (float)(std::log(10000.0) / (half - 1));
    for (int t = 0; t < 13; t++)
        for (int i = 0; i < half; i++) {
            const float a = (float)t * std::exp(-inc * (float)i);
            pos[t * ed + i] = to_bf16_bits(std::sin(a));
            pos[t * ed + half + i] = to_bf16_bits(std::cos(a));
        }
    // conv3 writes [chunk][t=ox][f=oy][c] so conv_out reads one contiguous row per (chunk, t).
    std::vector<int> map3(kChunkBatch * 208);
    for (int c = 0; c < kChunkBatch; c++)
        for (int oy = 0; oy < 16; oy++)
            for (int ox = 0; ox < 13; ox++) map3[c * 208 + oy * 13 + ox] = c * 208 + ox * 16 + oy;

    HIP_OK(hipMalloc(&tables, tab.size() * 4));
    HIP_OK(hipMalloc(&filters, fil.size() * 4));
    HIP_OK(hipMalloc(&inv_freq, 64 * 4));
    HIP_OK(hipMalloc(&pos13, pos.size() * 2));
    HIP_OK(hipMalloc(&conv3_map, map3.size() * 4));
    HIP_OK(hipMalloc(&gmax, 4));
    HIP_OK(hipMalloc(&argkeys, kMaxRows * 8));
    HIP_OK(hipMalloc(&ids_one, 8 * 4));
    HIP_OK(hipHostMalloc(&host_tok, 8 * 4, hipHostMallocDefault));
    HIP_OK(hipMemcpyAsync(tables, tab.data(), tab.size() * 4, hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(filters, fil.data(), fil.size() * 4, hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(inv_freq, inv.data(), 64 * 4, hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(pos13, pos.data(), pos.size() * 2, hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(conv3_map, map3.data(), map3.size() * 4, hipMemcpyHostToDevice, s));
    HIP_OK(hipStreamSynchronize(s));
    const double weights_ms = ms_since(t_read);
    const auto t_prep = Clock::now();
    prepare_decoder();
    HIP_OK(hipStreamSynchronize(s));
    char msg[256];
    std::snprintf(msg, sizeof msg,
                  "{\"hip_init_ms\":%.1f,\"alloc_ms\":%.1f,\"weights_ms\":%.1f,\"draft_prep_ms\":%.1f,"
                  "\"load_ms\":%.1f,\"weights_gib\":%.2f}",
                  hip_ms, alloc_ms, weights_ms, ms_since(t_prep), ms_since(t0), dev_bytes / 1073741824.0);
    stats = msg;
}

void QasrEngine::encoder(int T, const AttnTile* tiles, int n_tiles) {
    const int D = cfg.enc_d, F = cfg.enc_ffn, AO = cfg.enc_out, DD = cfg.dec_d;
    bf16* x = buf<bf16>("ex", (size_t)T * D);
    bf16* h = buf<bf16>("eh", (size_t)T * D);
    bf16* qkv = buf<bf16>("eqkv", (size_t)T * 3 * D);
    bf16* at = buf<bf16>("eattn", (size_t)T * D);
    bf16* ff = buf<bf16>("ef", (size_t)T * F);
    for (int l = 0; l < cfg.enc_layers; l++) {
        const std::string p = "enc." + std::to_string(l) + ".";
        layer_norm(s, x, w(p + "ln1.w"), w(p + "ln1.b"), h, T, D);
        Epi e;
        e.bias = w(p + "qkv.b"); e.out = qkv; e.ldo = 3 * D;
        gemm(s, h, D, w(p + "qkv.w"), T, 3 * D, D, e);
        AttnArgs a{qkv, 3 * D, 64, qkv + D, 3 * D, 64, qkv + 2 * D, 3 * D, 64, at, D, 64, 1, tiles, n_tiles, cfg.enc_heads, 0, 0,
                   0.125f};
        attention(s, a, 64);
        e = Epi{}; e.bias = w(p + "out.b"); e.res = x; e.res_ld = D; e.out = x; e.ldo = D;
        gemm(s, at, D, w(p + "out.w"), T, D, D, e);
        layer_norm(s, x, w(p + "ln2.w"), w(p + "ln2.b"), h, T, D);
        e = Epi{}; e.bias = w(p + "fc1.b"); e.act = ACT_GELU; e.out = ff; e.ldo = F;
        gemm(s, h, D, w(p + "fc1.w"), T, F, D, e);
        e = Epi{}; e.bias = w(p + "fc2.b"); e.res = x; e.res_ld = D; e.out = x; e.ldo = D;
        gemm(s, ff, F, w(p + "fc2.w"), T, D, F, e);
    }
    layer_norm(s, x, w("enc.ln_post.w"), w("enc.ln_post.b"), h, T, D);
    Epi e;
    e.bias = w("proj.1.b"); e.act = ACT_GELU; e.out = at; e.ldo = D;
    gemm(s, h, D, w("proj.1.w"), T, D, D, e);
    bf16* aemb = buf<bf16>("aemb", (size_t)T * DD);
    if (!cfg.adapter) {
        e = Epi{}; e.bias = w("proj.2.b"); e.out = aemb; e.ldo = AO;
        gemm(s, at, D, w("proj.2.w"), T, AO, D, e);
        return;
    }
    bf16* aout = buf<bf16>("aout", (size_t)T * AO);
    bf16* ah = buf<bf16>("ah", (size_t)T * AO);
    bf16* a1 = buf<bf16>("a1", (size_t)T * DD);
    e = Epi{}; e.bias = w("proj.2.b"); e.out = aout; e.ldo = AO;
    gemm(s, at, D, w("proj.2.w"), T, AO, D, e);
    layer_norm(s, aout, w("adapter.norm.w"), w("adapter.norm.b"), ah, T, AO);
    e = Epi{}; e.bias = w("adapter.1.b"); e.act = ACT_GELU; e.out = a1; e.ldo = DD;
    gemm(s, ah, AO, w("adapter.1.w"), T, DD, AO, e);
    e = Epi{}; e.bias = w("adapter.2.b"); e.out = aemb; e.ldo = DD;
    gemm(s, a1, DD, w("adapter.2.w"), T, DD, DD, e);
}

// Per-layer pointers, the static decode tile table and the 4-bit draft copy of the decoder.
void QasrEngine::prepare_decoder() {
    layers.resize(cfg.dec_layers);
    for (int l = 0; l < cfg.dec_layers; l++) {
        const std::string p = "dec." + std::to_string(l) + ".";
        Layer& L = layers[l];
        L = Layer{};
        L.ln1 = w(p + "ln1");
        L.ln2 = w(p + "ln2");
        if (cfg.qk_norm) { L.qn = w(p + "qnorm"); L.kn = w(p + "knorm"); }
        if (cfg.q4) {
            L.q_qkv = packed_q4(p + "qkv"); L.q_o = packed_q4(p + "o");
            L.q_gu = packed_q4(p + "gateup"); L.q_down = packed_q4(p + "down");
        } else {
            L.qkv = w(p + "qkv"); L.o = w(p + "o"); L.gu = w(p + "gateup"); L.down = w(p + "down");
        }
    }
    if (cfg.q4) {
        q_head = packed_q4("dec.head");
        q_embed = packed_q4("dec.embed");
        draft_len = 0;
    }
    std::vector<AttnTile> tiles((size_t)kMaxRows * kMaxSplits);
    for (int m = 1; m <= kMaxRows; m++)
        for (int sp = 0; sp < kMaxSplits; sp++)
            tiles[(size_t)(m - 1) * kMaxSplits + sp] = {0, m, sp * kSplitKeys, (sp + 1) * kSplitKeys};
    HIP_OK(hipMalloc(&dec_tiles, tiles.size() * sizeof(AttnTile)));
    HIP_OK(hipMemcpy(dec_tiles, tiles.data(), tiles.size() * sizeof(AttnTile), hipMemcpyHostToDevice));
    HIP_OK(hipMalloc(&part_o, (size_t)kMaxSplits * cfg.q_heads * 16 * 128 * 4));
    HIP_OK(hipMalloc(&part_ml, (size_t)kMaxSplits * cfg.q_heads * 16 * 2 * 4));
    HIP_OK(hipMalloc(&ids_round, kMaxRows * 4));
    HIP_OK(hipMalloc(&ids_ver, kMaxRows * 4));
    HIP_OK(hipMalloc(&pair, kMaxRows * 2 * 4));
    if (const char* v = std::getenv("QASR_DRAFT"); v && !cfg.q4) draft_len = std::max(0, std::min(kMaxRows - 1, std::atoi(v)));
    if (!draft_len) return;

    struct Mat { Q4* dst; const bf16* src; int N, K; };
    std::vector<Mat> mats;
    const int D = cfg.dec_d;
    for (auto& L : layers) {
        mats.push_back({&L.q_qkv, L.qkv, cfg.qkv_w(), D});
        mats.push_back({&L.q_o, L.o, D, cfg.q_w()});
        mats.push_back({&L.q_gu, L.gu, 2 * cfg.dec_ffn, D});
        mats.push_back({&L.q_down, L.down, D, cfg.dec_ffn});
    }
    mats.push_back({&q_head, w("dec.embed"), cfg.vocab, D});
    size_t bytes = 0;
    for (auto& m : mats) bytes += (size_t)m.N * m.K / 2 + ((size_t)m.N * (m.K / 32) * 2 + 255) / 256 * 256;
    HIP_OK(hipMalloc(&q4_mem, bytes));
    char* at = (char*)q4_mem;
    for (auto& m : mats) {
        uint4* q = (uint4*)at;
        at += (size_t)m.N * m.K / 2;
        f16* sc = (f16*)at;
        at += ((size_t)m.N * (m.K / 32) * 2 + 255) / 256 * 256;
        quantize_q4(s, m.src, m.N, m.K, q, sc);
        *m.dst = Q4{q, sc};
    }
}

void QasrEngine::decoder_layers(int M, int pos0, const AttnTile* tiles, int n_tiles, bool draft) {
    const int D = cfg.dec_d, QKV = cfg.qkv_w(), QW = cfg.q_w(), KVW = cfg.kv_w(), FF = cfg.dec_ffn;
    bf16* x = buf<bf16>("dx", (size_t)M * D);
    bf16* h = buf<bf16>("dh", (size_t)M * D);
    bf16* qkv = buf<bf16>("dqkv", (size_t)M * QKV);
    bf16* at = buf<bf16>("dattn", (size_t)M * QW);
    bf16* gu = buf<bf16>("dgu", (size_t)M * 2 * FF);
    bf16* act = buf<bf16>("dact", (size_t)M * FF);
    bf16* kv = (bf16*)scratch.at("kv").first;
    const size_t layer_kv = kv_pos * KVW;  // elements per K (or V) plane
    const bool split = M <= kMaxRows;
    const int splits = cdiv(pos0 + M, kSplitKeys);
    // One row multiplies the 4-bit weights directly; more rows multiply them unpacked to bf16, a matrix at a time.
    auto mm = [&](const bf16* a, int lda, const bf16* wt, const Q4& wq, int N, int K, const Epi& e) {
        if (draft || (cfg.q4 && M == 1)) gemv_q4(s, a, wq, N, K, e);
        else if (cfg.q4) {
            bf16* unpacked = buf<bf16>("wdq", (size_t)N * K);
            dequant_q4(s, wq, N, K, unpacked);
            gemm(s, a, lda, unpacked, M, N, K, e);
        } else gemm(s, a, lda, wt, M, N, K, e);
    };
    for (int l = 0; l < cfg.dec_layers; l++) {
        const Layer& Ly = layers[l];
        bf16* kc = kv + (size_t)l * 2 * layer_kv;
        bf16* vc = kc + layer_kv;
        const bool pf = M > kMaxRows;
        if (pf) tick("p.other");
        rms_norm(s, x, D, Ly.ln1, h, M, D, cfg.norm_groups);
        if (pf) tick("p.norm");
        Epi e;
        e.out = qkv; e.ldo = QKV;
        mm(h, D, Ly.qkv, Ly.q_qkv, QKV, D, e);
        if (pf) tick("p.qkv");
        qk_norm_rope(s, qkv, Ly.qn, Ly.kn, kc, vc, M, pos0, inv_freq, cfg.q_heads, cfg.kv_heads);
        if (pf) tick("p.rope");
        AttnArgs a{qkv, QKV, 128, kc, KVW, 128, vc, KVW, 128, at, QW, 128, cfg.q_heads / cfg.kv_heads, tiles, n_tiles,
                   cfg.q_heads, 1, pos0, 0.08838834764831845f};
        if (split) {
            a.tiles = dec_tiles + (size_t)(M - 1) * kMaxSplits;
            a.n_tiles = splits;
            a.part_o = part_o;
            a.part_ml = part_ml;
            attention(s, a, 128);
            attention_combine(s, a, M, splits);
        } else {
            attention(s, a, 128);
        }
        if (pf) tick("p.attn");
        e = Epi{}; e.res = x; e.res_ld = D; e.out = x; e.ldo = D;
        mm(at, QW, Ly.o, Ly.q_o, D, QW, e);
        if (pf) tick("p.o");
        rms_norm(s, x, D, Ly.ln2, h, M, D, cfg.norm_groups);
        if (pf) tick("p.norm");
        e = Epi{}; e.out = gu; e.ldo = 2 * FF;
        mm(h, D, Ly.gu, Ly.q_gu, 2 * FF, D, e);
        if (pf) tick("p.gateup");
        silu_mul(s, gu, act, M, FF);
        if (pf) tick("p.silu");
        e = Epi{}; e.res = x; e.res_ld = D; e.out = x; e.ldo = D;
        mm(act, FF, Ly.down, Ly.q_down, D, FF, e);
        if (pf) tick("p.down");
    }
}

// Final norm and lm_head over M rows of x; the argmax ids land in ids (device).
// A 4-bit head takes one row at a time, which is all a model without a draft asks of it.
void QasrEngine::logits_argmax(const bf16* x, int M, int* ids, bool draft) {
    const int D = cfg.dec_d;
    bf16* dh = buf<bf16>("dh", (size_t)M * D);
    rms_norm(s, x, D, w("dec.norm"), dh, M, D, cfg.norm_groups);
    HIP_OK(hipMemsetAsync(argkeys, 0, (size_t)M * 8, s));
    Epi e;
    e.argmax = argkeys;
    if (draft || cfg.q4) gemv_q4(s, dh, q_head, cfg.vocab, D, e);
    else gemm(s, dh, D, w("dec.embed"), M, cfg.vocab, D, e);
    argmax_ids(s, argkeys, ids, M);
    if (!draft && dot_id >= 0 && question_id >= 0) pair_logits(s, dh, M, w("dec.embed"), dot_id, question_id, pair, D);
}

// Greedy decoding after the prefill. With a draft, each round guesses draft_len tokens with the
// 4-bit copy, then one bf16 pass over [last, drafts...] checks them all. Verify rows are computed
// exactly like single steps (same GEMV order, masked keys add nothing), so the output is the
// bf16 model's own greedy output either way; the draft only changes how fast it arrives.
std::vector<int> QasrEngine::generate(const bf16* last_row, int L, int max_new) {
    bf16* dx = buf<bf16>("dx", (size_t)kMaxRows * cfg.dec_d);
    std::vector<int> out;
    rounds = accepted = 0;
    question_probability = -1.f;
    logits_argmax(last_row, 1, ids_round);
    int pos = L;
    int host[2 * kMaxRows];
    float pairs[2 * kMaxRows];
    HIP_OK(hipMemcpyAsync(host, ids_round, 4, hipMemcpyDeviceToHost, s));
    HIP_OK(hipMemcpyAsync(pairs, pair, 2 * 4, hipMemcpyDeviceToHost, s));
    HIP_OK(hipStreamSynchronize(s));
    auto finished = [&](int t) { return t == cfg.eos1 || t == cfg.eos2 || (int)out.size() >= max_new; };
    // Each kept token sets the odds when it is punctuation and clears them otherwise, so after the
    // loop they belong to the last token.
    auto keep = [&](int t, const float* logits) {
        out.push_back(t);
        question_probability = t == dot_id || t == question_id
            ? 1.f / (1.f + std::exp(logits[0] - logits[1])) : -1.f;
    };
    if (finished(host[0])) return out;
    keep(host[0], pairs);
    while ((int)out.size() < max_new) {
        const int k = std::min(draft_len, max_new - (int)out.size());
        for (int i = 0; i < k; i++) {
            embed_rows(ids_round + i, nullptr, 0, dx, 1);
            decoder_layers(1, pos + i, nullptr, 0, true);
            logits_argmax(dx, 1, ids_round + i + 1, true);
        }
        embed_rows(ids_round, nullptr, 0, dx, k + 1);
        decoder_layers(k + 1, pos, nullptr, 0);
        logits_argmax(dx, k + 1, ids_ver);
        HIP_OK(hipMemcpyAsync(host, ids_round, (k + 1) * 4, hipMemcpyDeviceToHost, s));
        HIP_OK(hipMemcpyAsync(host + kMaxRows, ids_ver, (k + 1) * 4, hipMemcpyDeviceToHost, s));
        HIP_OK(hipMemcpyAsync(pairs, pair, (k + 1) * 2 * 4, hipMemcpyDeviceToHost, s));
        HIP_OK(hipStreamSynchronize(s));
        int a = 0;
        while (a < k && host[1 + a] == host[kMaxRows + a]) a++;
        rounds++;
        accepted += a;
        // Drafts 1..a equal the verified tokens; the verified token after them comes free.
        for (int i = 0; i <= a; i++) {
            const int t = host[kMaxRows + i];
            if (finished(t)) return out;
            keep(t, pairs + 2 * i);
        }
        pos += a + 1;
        HIP_OK(hipMemcpyAsync(ids_round, ids_ver + a, 4, hipMemcpyDeviceToDevice, s));
    }
    return out;
}

std::string QasrEngine::detokenize(const std::vector<int>& ids) {
    std::string bytes;
    for (int id : ids)
        if (id >= 0 && id < cfg.vocab) bytes.append((const char*)vocab.data() + vocab_off[id], vocab_off[id + 1] - vocab_off[id]);
    std::u32string text = strip(utf8_decode(bytes));
    if (text.empty()) return "";
    text = fix_pattern_repeats(fix_char_repeats(text, 20), 20);
    static const std::u32string marker = U"<asr_text>";
    size_t at = text.find(marker);
    if (at != std::u32string::npos) text = text.substr(at + marker.size());
    return utf8_encode(strip(text));
}

std::string QasrEngine::transcribe(const float* pcm, int n, const std::vector<int>& pre, const std::vector<int>& suf) {
    const auto t0 = Clock::now();
    const int n_eff = std::max(n, 8000);
    const int frames = n_eff / 160, fpad = cdiv(frames, 100) * 100, chunks = fpad / 100;
    const int max_new = (int)(n / 16000.0 * 8) + 32;

    // Packed encoder positions: each chunk keeps (len-1)/2+1 three times over of its 13 slots.
    std::vector<int> out_map(chunks * 13, -1);
    int T = 0;
    for (int c = 0; c < chunks; c++) {
        int len = std::min(100, frames - c * 100);
        for (int k = 0; k < 3; k++) len = len > 0 ? (len - 1) / 2 + 1 : 0;
        for (int t = 0; t < len; t++) out_map[c * 13 + t] = T++;
    }
    std::vector<AttnTile> tiles;
    for (int ws = 0; ws < T; ws += 104) {
        const int we = std::min(T, ws + 104);
        for (int q = ws; q < we; q += 16) tiles.push_back({q, std::min(q + 16, we), ws, we});
    }
    const int n_enc_tiles = (int)tiles.size();
    const int L = (int)pre.size() + T + (int)suf.size();
    // Prefill tiles per row chunk, query rows relative to the chunk; causal masking uses pos_off.
    std::vector<int> chunk_tiles;
    for (int r0 = 0; r0 < L; r0 += kPrefillRows) {
        const int rows_here = std::min(kPrefillRows, L - r0);
        chunk_tiles.push_back((int)tiles.size());
        for (int q = 0; q < rows_here; q += 16) tiles.push_back({q, std::min(q + 16, rows_here), 0, INT_MAX});
    }
    std::vector<int> ids(pre);
    ids.resize(pre.size() + T, -1);
    ids.insert(ids.end(), suf.begin(), suf.end());

    // Scratch, sized before any kernel runs so growing it never stalls mid-stream.
    float* pcm_d = buf<float>("pcm", n_eff);
    float* mel_scratch = buf<float>("mel", (size_t)frames * 128);
    bf16* melT = buf<bf16>("melT", (size_t)128 * fpad);
    const int batch = std::min(chunks, kChunkBatch);
    bf16* c1 = buf<bf16>("c1", (size_t)batch * 3200 * 480);
    bf16* c2 = buf<bf16>("c2", (size_t)batch * 800 * 480);
    bf16* c3 = buf<bf16>("c3", (size_t)batch * 208 * 480);
    int* map_d = buf<int>("outmap", out_map.size());
    AttnTile* tiles_d = buf<AttnTile>("tiles", tiles.size());
    int* ids_d = buf<int>("ids", ids.size());
    const int ED = cfg.enc_d, DD = cfg.dec_d, KVW = cfg.kv_w();
    buf<bf16>("ex", (size_t)T * ED);
    buf<bf16>("aemb", (size_t)T * DD);
    if (cfg.adapter) {
        buf<bf16>("aout", (size_t)T * cfg.enc_out);
        buf<bf16>("ah", (size_t)T * cfg.enc_out);
        buf<bf16>("a1", (size_t)T * DD);
    }
    const size_t rows = std::max(std::min(L, kPrefillRows), kMaxRows);
    buf<bf16>("dx", rows * DD);
    if ((size_t)L + max_new + kMaxRows + 1 > (size_t)kMaxSplits * kSplitKeys) throw std::runtime_error("audio too long");
    // Sized once for the whole call: growing it mid-decode needs old and new copies at once, and when
    // VRAM is short Windows places the new one in system memory, which halves decode speed.
    buf<bf16>("kv", ((size_t)L + max_new + kMaxRows + 1) * KVW * 2 * cfg.dec_layers);
    kv_pos = scratch.at("kv").second / (2 * (size_t)cfg.dec_layers * KVW * sizeof(bf16));
    for (const char* name : {"eh", "eattn"}) buf<bf16>(name, (size_t)T * ED);
    buf<bf16>("eqkv", (size_t)T * 3 * ED);
    buf<bf16>("ef", (size_t)T * cfg.enc_ffn);
    buf<bf16>("dh", rows * DD);
    buf<bf16>("dqkv", rows * cfg.qkv_w());
    buf<bf16>("dattn", rows * cfg.q_w());
    buf<bf16>("dgu", rows * 2 * cfg.dec_ffn);
    buf<bf16>("dact", rows * cfg.dec_ffn);
    if (cfg.q4)
        buf<bf16>("wdq", std::max({(size_t)cfg.qkv_w() * DD, (size_t)DD * cfg.q_w(), (size_t)2 * cfg.dec_ffn * DD}));

    std::vector<float> padded(pcm, pcm + n);
    padded.resize(n_eff, 0.f);
    HIP_OK(hipMemcpyAsync(pcm_d, padded.data(), n_eff * 4, hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(map_d, out_map.data(), out_map.size() * 4, hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(tiles_d, tiles.data(), tiles.size() * sizeof(AttnTile), hipMemcpyHostToDevice, s));
    HIP_OK(hipMemcpyAsync(ids_d, ids.data(), ids.size() * 4, hipMemcpyHostToDevice, s));

    // Phase boundaries sync the stream: a few microseconds each, and the times are then exact.
    Clock::time_point mark[5];
    auto phase = [&](int i) { HIP_OK(hipStreamSynchronize(s)); mark[i] = Clock::now(); };
    phase(0);
    log_mel(s, pcm_d, n_eff, frames, fpad, tables, filters, mel_scratch, gmax, melT);
    bf16* x = buf<bf16>("ex", (size_t)T * ED);
    for (int c0 = 0; c0 < chunks; c0 += kChunkBatch) {
        const int nb = std::min(kChunkBatch, chunks - c0);
        conv1(s, melT, fpad, c0, nb, w("enc.conv1.w"), w("enc.conv1.b"), c1);
        Epi e;
        e.bias = w("enc.conv2.b"); e.act = ACT_GELU; e.out = c2; e.ldo = 480;
        conv3x3_480(s, c1, nb, 64, 50, w("enc.conv2.w"), 480, e);
        e = Epi{}; e.bias = w("enc.conv3.b"); e.act = ACT_GELU; e.row_map = conv3_map; e.out = c3; e.ldo = 480;
        conv3x3_480(s, c2, nb, 32, 25, w("enc.conv3.w"), 480, e);
        e = Epi{}; e.res = pos13; e.res_ld = ED; e.res_mod = 13; e.row_map = map_d + c0 * 13; e.out = x; e.ldo = ED;
        gemm(s, c3, 7680, w("enc.conv_out.w"), nb * 13, ED, 7680, e);
    }
    phase(1);
    encoder(T, tiles_d, n_enc_tiles);
    phase(2);

    bf16* dx = buf<bf16>("dx", rows * DD);
    int last_rows = 0;
    for (int c = 0, r0 = 0; r0 < L; c++, r0 += kPrefillRows) {
        last_rows = std::min(kPrefillRows, L - r0);
        embed_rows(ids_d + r0, buf<bf16>("aemb", 1), (int)pre.size() - r0, dx, last_rows);
        if (last_rows <= kMaxRows) decoder_layers(last_rows, r0, nullptr, 0);
        else decoder_layers(last_rows, r0, tiles_d + chunk_tiles[c], cdiv(last_rows, 16));
    }
    phase(3);
    const std::vector<int> out = generate(dx + (size_t)(last_rows - 1) * DD, L, max_new);
    phase(4);
    double part[4];
    for (int i = 0; i < 4; i++) part[i] = std::chrono::duration<double, std::milli>(mark[i + 1] - mark[i]).count();

    if (scratch.at("kv").second > kKvKeepBytes) {
        HIP_OK(hipFree(scratch.at("kv").first));
        scratch.erase("kv");
        kv_pos = 0;
    }
    const std::string text = detokenize(out);
    char msg[320];
    std::snprintf(msg, sizeof msg,
                  "{\"stem_ms\":%.2f,\"encoder_ms\":%.2f,\"prefill_ms\":%.2f,\"decode_ms\":%.2f,\"total_ms\":%.2f,"
                  "\"audio_tokens\":%d,\"tokens\":%zu,\"rounds\":%d,\"accepted\":%d}",
                  part[0], part[1], part[2], part[3], ms_since(t0), T, out.size(), rounds, accepted);
    stats = msg;
    if (profile) {
        stats.pop_back();
        for (auto& p : prof) stats += ",\"" + p.first + "\":" + std::to_string(p.second);
        stats += "}";
        prof.clear();
    }
    return text;
}

extern "C" {

QASR_API QasrEngine* qasr_load(const char* path, char* err, int err_len) {
    auto* e = new QasrEngine();
    try {
        e->load(path);
        return e;
    } catch (const std::exception& x) {
        if (err && err_len > 0) std::snprintf(err, err_len, "%s", x.what());
        qasr_free(e);
        return nullptr;
    }
}

QASR_API int qasr_transcribe(QasrEngine* e, const float* pcm, int samples, char* text, int text_len) {
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

QASR_API int qasr_transcribe_prompted(QasrEngine* e, const float* pcm, int samples, const int* prefix, int prefix_len,
                                      const int* suffix, int suffix_len, char* text, int text_len) {
    try {
        const std::vector<int> pre(prefix, prefix + prefix_len), suf(suffix, suffix + suffix_len);
        const std::string t = e->transcribe(pcm, samples, pre, suf);
        if ((int)t.size() >= text_len) throw std::runtime_error("text buffer too small");
        std::memcpy(text, t.c_str(), t.size() + 1);
        return (int)t.size();
    } catch (const std::exception& x) {
        if (text && text_len > 0) std::snprintf(text, text_len, "%s", x.what());
        return -1;
    }
}

QASR_API const char* qasr_stats(QasrEngine* e) { return e->stats.c_str(); }

QASR_API float qasr_question_probability(QasrEngine* e) { return e->question_probability; }

QASR_API void qasr_free(QasrEngine* e) {
    if (!e) return;
    (void)hipDeviceSynchronize();
    for (auto& kv : e->scratch) (void)hipFree(kv.second.first);
    void* devs[] = {e->dev, e->tables, e->filters, e->inv_freq, e->pos13, e->conv3_map, e->gmax, e->argkeys, e->ids_one,
                    e->dec_tiles, e->part_o, e->part_ml, e->ids_round, e->ids_ver, e->q4_mem, e->pair};
    for (void* p : devs) if (p) (void)hipFree(p);
    if (e->host_tok) (void)hipHostFree(e->host_tok);
    if (e->s) (void)hipStreamDestroy(e->s);
    delete e;
}

}  // extern "C"
