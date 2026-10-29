// Device kernels particular to Parakeet TDT on RDNA 4 (gfx1200, wave32, WMMA bf16).
#include "pk_kernels.h"

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

#define NEG_INF (-__builtin_huge_valf())

namespace {

constexpr int MELS = 128, BINS = 257, NFFT = 512, HOP = 160;
constexpr int D = 1024, HEADS = 8, DK = 128, QKV = 3072;
constexpr int PRED = 640, VOCAB_OUT = 8198, BLANK = 8192, DURATIONS = 5;

__device__ __forceinline__ float f(bf16 x) { return (float)x; }
__device__ __forceinline__ bf16 b(float x) { return (bf16)x; }
__device__ __forceinline__ float sigm(float x) { return 1.f / (1.f + expf(-x)); }
__device__ __forceinline__ f32x8 wmma(bf16x8 a, bf16x8 bb, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, bb, c);
}
__device__ __forceinline__ bf16x8 as_bf8(uint4 v) { return __builtin_bit_cast(bf16x8, v); }
__device__ __forceinline__ float wave_sum(float v) {
    for (int m = 16; m; m >>= 1) v += __shfl_xor(v, m, 32);
    return v;
}
__device__ float block_sum(float v, float* red) {
    v = wave_sum(v);
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, waves = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) red[wave] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < waves; i++) t += red[i];
    return t;
}
int cdiv(int a, int c) { return (a + c - 1) / c; }

// ---- features ----

// One frame per block. The padded signal is the pre-emphasised one, reflected at both ends (torch.stft, center).
__global__ __launch_bounds__(256) void log_mel_kernel(const float* __restrict__ pcm, int n, const float* __restrict__ win,
                                                      const float* __restrict__ cs, const float* __restrict__ fb, float* out) {
    __shared__ float x[NFFT], pw[BINS];
    const int t = blockIdx.x, tid = threadIdx.x;
    for (int i = tid; i < NFFT; i += 256) {
        int q = t * HOP + i - NFFT / 2;
        if (q < 0) q = -q;
        if (q >= n) q = 2 * (n - 1) - q;
        q = max(0, min(n - 1, q));
        const float y = q == 0 ? pcm[0] : pcm[q] - 0.97f * pcm[q - 1];
        x[i] = y * win[i];
    }
    __syncthreads();
    for (int k = tid; k < BINS; k += 256) {
        float re = 0.f, im = 0.f;
        int idx = 0;
        for (int i = 0; i < NFFT; i++) {
            re = fmaf(x[i], cs[idx], re);
            im = fmaf(x[i], cs[NFFT + idx], im);
            idx = (idx + k) & (NFFT - 1);
        }
        pw[k] = re * re + im * im;
    }
    __syncthreads();
    for (int m = tid; m < MELS; m += 256) {
        float acc = 0.f;
        for (int k = 0; k < BINS; k++) acc = fmaf(fb[m * BINS + k], pw[k], acc);
        out[(size_t)t * MELS + m] = logf(acc + 5.960464e-8f);
    }
}

// One band per block: (x - mean) / (std + 1e-5) with the unbiased std, as bf16 [frames][128].
__global__ __launch_bounds__(256) void normalize_kernel(const float* __restrict__ mel, int frames, bf16* out) {
    __shared__ float red[8];
    const int m = blockIdx.x;
    float s = 0.f;
    for (int t = threadIdx.x; t < frames; t += 256) s += mel[(size_t)t * MELS + m];
    const float mean = block_sum(s, red) / frames;
    float q = 0.f;
    for (int t = threadIdx.x; t < frames; t += 256) {
        const float d = mel[(size_t)t * MELS + m] - mean;
        q += d * d;
    }
    const float var = block_sum(q, red) / max(frames - 1, 1);
    const float inv = 1.f / (sqrtf(var) + 1e-5f);
    for (int t = threadIdx.x; t < frames; t += 256) out[(size_t)t * MELS + m] = b((mel[(size_t)t * MELS + m] - mean) * inv);
}

// ---- subsampling ----

__global__ void conv0_kernel(const bf16* __restrict__ in, int t_in, const bf16* __restrict__ w, const bf16* __restrict__ bias,
                             bf16* out, int t_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t_out * 64 * 256) return;
    const int c = idx & 255, p = idx >> 8, fo = p % 64, to = p / 64;
    float acc = f(bias[c]);
    for (int ky = 0; ky < 3; ky++) {
        const int ti = 2 * to - 1 + ky;
        if (ti < 0 || ti >= t_in) continue;
        for (int kx = 0; kx < 3; kx++) {
            const int fi = 2 * fo - 1 + kx;
            if (fi < 0 || fi >= MELS) continue;
            acc = fmaf(f(w[c * 9 + ky * 3 + kx]), f(in[(size_t)ti * MELS + fi]), acc);
        }
    }
    out[idx] = b(fmaxf(acc, 0.f));
}

__global__ void dwconv_kernel(const bf16* __restrict__ in, int t_in, int f_in, const bf16* __restrict__ w,
                              const bf16* __restrict__ bias, bf16* out, int t_out, int f_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= t_out * f_out * 256) return;
    const int c = idx & 255, p = idx >> 8, fo = p % f_out, to = p / f_out;
    float acc = f(bias[c]);
    for (int ky = 0; ky < 3; ky++) {
        const int ti = 2 * to - 1 + ky;
        if (ti < 0 || ti >= t_in) continue;
        for (int kx = 0; kx < 3; kx++) {
            const int fi = 2 * fo - 1 + kx;
            if (fi < 0 || fi >= f_in) continue;
            acc = fmaf(f(w[c * 9 + ky * 3 + kx]), f(in[((size_t)ti * f_in + fi) * 256 + c]), acc);
        }
    }
    out[idx] = b(acc);
}

// ---- relative-position attention: one wave per (16 query rows, head), online softmax over 16-key tiles ----
//
// The positional term for query i and key j uses p row t - 1 - i + j. For one 16 x 16 tile those rows span 31
// consecutive ones, so two WMMA tiles against them cover it, and a diagonal read from LDS picks each entry.
__global__ __launch_bounds__(32) void relpos_attn_kernel(const bf16* __restrict__ qkv, const bf16* __restrict__ p,
                                                         const bf16* __restrict__ u, const bf16* __restrict__ vb, int T,
                                                         bf16* out) {
    constexpr int KS = DK / 16;
    __shared__ __attribute__((aligned(16))) bf16 Ps[16][24];
    __shared__ __attribute__((aligned(16))) bf16 Vs[16][DK + 8];
    __shared__ float Bd[16][33];
    const int lane = threadIdx.x, h = blockIdx.y, q0 = blockIdx.x * 16, hl = lane >> 4, col = lane & 15;
    const int npos = 2 * T - 1;
    bf16x8 qu[KS], qv[KS];
    {
        const int qrow = q0 + col;
        for (int ks = 0; ks < KS; ks++)
            for (int e = 0; e < 8; e++) {
                const int d = ks * 16 + hl * 8 + e;
                const float q = qrow < T ? f(qkv[(size_t)qrow * QKV + h * DK + d]) : 0.f;
                qu[ks][e] = qrow < T ? b(q + f(u[h * DK + d])) : b(0.f);
                qv[ks][e] = qrow < T ? b(q + f(vb[h * DK + d])) : b(0.f);
            }
    }
    f32x8 o[KS];
    for (int nt = 0; nt < KS; nt++) o[nt] = f32x8{};
    float m[8], l[8];
    for (int i = 0; i < 8; i++) { m[i] = NEG_INF; l[i] = 0.f; }
    const float scale = 0.08838834764831845f;  // 1 / sqrt(128)
    for (int kb = 0; kb < T; kb += 16) {
        const int key = kb + col;
        f32x8 sc = f32x8{}, b0 = f32x8{}, b1 = f32x8{};
        const int base = T - 1 - (q0 + 15) + kb;
        const int r0 = base + col, r1 = base + 16 + col;
        for (int ks = 0; ks < KS; ks++) {
            const int off = h * DK + ks * 16 + hl * 8;
            const bf16x8 kf = as_bf8(key < T ? *(const uint4*)(qkv + (size_t)key * QKV + D + off) : uint4{0, 0, 0, 0});
            sc = wmma(qu[ks], kf, sc);
            const bf16x8 p0 = as_bf8(r0 >= 0 && r0 < npos ? *(const uint4*)(p + (size_t)r0 * D + off) : uint4{0, 0, 0, 0});
            const bf16x8 p1 = as_bf8(r1 >= 0 && r1 < npos ? *(const uint4*)(p + (size_t)r1 * D + off) : uint4{0, 0, 0, 0});
            b0 = wmma(qv[ks], p0, b0);
            b1 = wmma(qv[ks], p1, b1);
        }
        for (int i = 0; i < 8; i++) {
            Bd[hl * 8 + i][col] = b0[i];
            Bd[hl * 8 + i][16 + col] = b1[i];
        }
        for (int j = lane; j < 16 * DK / 8; j += 32) {
            const int kr = j / (DK / 8), dv = (j - kr * (DK / 8)) * 8, kk = kb + kr;
            *(uint4*)&Vs[kr][dv] = kk < T ? *(const uint4*)(qkv + (size_t)kk * QKV + 2 * D + h * DK + dv) : uint4{0, 0, 0, 0};
        }
        __syncthreads();
        for (int i = 0; i < 8; i++) {
            const int li = hl * 8 + i, qr = q0 + li;
            const bool ok = key < T && qr < T;
            const float v = ok ? (sc[i] + Bd[li][col - li + 15]) * scale : NEG_INF;
            float mx = v;
            for (int x = 1; x < 16; x <<= 1) mx = fmaxf(mx, __shfl_xor(mx, x, 32));
            const float mn = fmaxf(m[i], mx);
            const float pr = mn == NEG_INF ? 0.f : expf(v - mn);
            const float alpha = mn == NEG_INF ? 1.f : expf(m[i] - mn);
            float ps = pr;
            for (int x = 1; x < 16; x <<= 1) ps += __shfl_xor(ps, x, 32);
            l[i] = l[i] * alpha + ps;
            m[i] = mn;
            for (int nt = 0; nt < KS; nt++) o[nt][i] *= alpha;
            Ps[li][col] = b(pr);
        }
        __syncthreads();
        const bf16x8 pf = as_bf8(*(const uint4*)&Ps[col][hl * 8]);
        for (int nt = 0; nt < KS; nt++) {
            bf16x8 vf;
            for (int j = 0; j < 8; j++) vf[j] = Vs[hl * 8 + j][nt * 16 + col];
            o[nt] = wmma(pf, vf, o[nt]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 8; i++) {
        const int row = q0 + hl * 8 + i;
        if (row >= T) continue;
        const float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
        for (int nt = 0; nt < KS; nt++) out[(size_t)row * D + h * DK + nt * 16 + col] = b(o[nt][i] * inv);
    }
}

// ---- conformer convolution module middle: GLU, depthwise conv (batch norm folded), SiLU ----

__global__ void glu_dwconv_silu_kernel(const bf16* __restrict__ g, int T, const float* __restrict__ w,
                                       const float* __restrict__ bias, bf16* out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * D) return;
    const int c = idx % D, t = idx / D;
    float acc = bias[c];
    for (int k = 0; k < 9; k++) {
        const int ti = t + k - 4;
        if (ti < 0 || ti >= T) continue;
        const float glu = f(b(f(g[(size_t)ti * 2 * D + c]) * sigm(f(g[(size_t)ti * 2 * D + D + c]))));
        acc = fmaf(w[c * 9 + k], glu, acc);
    }
    out[idx] = b(acc * sigm(acc));
}

// ---- TDT decoder ----

// One block per unit: its four gate rows against [x; h], then the cell update.
__global__ __launch_bounds__(128) void lstm_step_kernel(const bf16* __restrict__ x, const float* __restrict__ h,
                                                        const float* __restrict__ c, const bf16* __restrict__ w_ih,
                                                        const bf16* __restrict__ w_hh, const float* __restrict__ bias,
                                                        float* h_out, float* c_out, bf16* h_bf16) {
    __shared__ float red[4];
    const int unit = blockIdx.x;
    float gate[4];
    for (int g = 0; g < 4; g++) {
        const int row = g * PRED + unit;
        float acc = 0.f;
        for (int k = threadIdx.x; k < PRED; k += 128)
            acc = fmaf(f(w_ih[(size_t)row * PRED + k]), f(x[k]), fmaf(f(w_hh[(size_t)row * PRED + k]), h[k], acc));
        gate[g] = block_sum(acc, red) + bias[row];
    }
    if (threadIdx.x == 0) {
        const float cn = sigm(gate[1]) * c[unit] + sigm(gate[0]) * tanhf(gate[2]);
        const float hn = sigm(gate[3]) * tanhf(cn);
        c_out[unit] = cn;
        h_out[unit] = hn;
        h_bf16[unit] = b(hn);
    }
}

__global__ __launch_bounds__(128) void linear640_kernel(const float* __restrict__ h, const bf16* __restrict__ w,
                                                        const bf16* __restrict__ bias, float* out) {
    __shared__ float red[4];
    const int j = blockIdx.x;
    float acc = 0.f;
    for (int k = threadIdx.x; k < PRED; k += 128) acc = fmaf(f(w[(size_t)j * PRED + k]), f(b(h[k])), acc);
    acc = block_sum(acc, red);
    if (threadIdx.x == 0) out[j] = f(b(acc + f(bias[j])));
}

__global__ void joint_window_kernel(const bf16* __restrict__ enc, const float* __restrict__ pred, const int* __restrict__ tT,
                                    int rows, bf16* z) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * PRED) return;
    const int t = tT[0] + idx / PRED, j = idx % PRED;
    z[idx] = t < tT[1] ? b(fmaxf(f(b(f(enc[(size_t)t * PRED + j]) + pred[j])), 0.f)) : b(0.f);
}

__global__ __launch_bounds__(256) void token_duration_kernel(const bf16* __restrict__ logits, int* out) {
    __shared__ float bv[8];
    __shared__ int bi[8];
    const int row = blockIdx.x, lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const bf16* lr = logits + (size_t)row * VOCAB_OUT;
    float best = NEG_INF;
    int arg = 0;
    for (int k = threadIdx.x; k <= BLANK; k += 256) {
        const float v = f(lr[k]);
        if (v > best) { best = v; arg = k; }
    }
    for (int x = 16; x; x >>= 1) {
        const float ov = __shfl_xor(best, x, 32);
        const int oi = __shfl_xor(arg, x, 32);
        if (ov > best || (ov == best && oi < arg)) { best = ov; arg = oi; }
    }
    if (lane == 0) { bv[wave] = best; bi[wave] = arg; }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; w++)
            if (bv[w] > best || (bv[w] == best && bi[w] < arg)) { best = bv[w]; arg = bi[w]; }
        int dur = 0;
        float dbest = f(lr[BLANK + 1]);
        for (int d = 1; d < DURATIONS; d++)
            if (f(lr[BLANK + 1 + d]) > dbest) { dbest = f(lr[BLANK + 1 + d]); dur = d; }
        out[2 * row] = arg;
        out[2 * row + 1] = dur;
    }
}

__global__ void embed_kernel(const bf16* __restrict__ table, const int* __restrict__ label, bf16* x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < PRED) x[i] = table[(size_t)label[0] * PRED + i];
}

}  // namespace

void pk_log_mel(hipStream_t s, const float* pcm, int n, int frames, const float* window512, const float* cos_sin,
                const float* fb, float* out) {
    log_mel_kernel<<<frames, 256, 0, s>>>(pcm, n, window512, cos_sin, fb, out);
}

void pk_normalize(hipStream_t s, const float* mel, int frames, bf16* out) {
    normalize_kernel<<<MELS, 256, 0, s>>>(mel, frames, out);
}

void pk_conv0(hipStream_t s, const bf16* in, int t, const bf16* w, const bf16* bias, bf16* out) {
    const int t_out = (t - 1) / 2 + 1;
    conv0_kernel<<<cdiv(t_out * 64 * 256, 256), 256, 0, s>>>(in, t, w, bias, out, t_out);
}

void pk_dwconv(hipStream_t s, const bf16* in, int t, int f_in, const bf16* w, const bf16* bias, bf16* out) {
    const int t_out = (t - 1) / 2 + 1, f_out = (f_in - 1) / 2 + 1;
    dwconv_kernel<<<cdiv(t_out * f_out * 256, 256), 256, 0, s>>>(in, t, f_in, w, bias, out, t_out, f_out);
}

void pk_relpos_attention(hipStream_t s, const bf16* qkv, const bf16* p, const bf16* u, const bf16* v, int t, bf16* out) {
    relpos_attn_kernel<<<dim3(cdiv(t, 16), HEADS), 32, 0, s>>>(qkv, p, u, v, t, out);
}

void pk_glu_dwconv_silu(hipStream_t s, const bf16* g, int t, const float* w, const float* bias, bf16* out) {
    glu_dwconv_silu_kernel<<<cdiv(t * D, 256), 256, 0, s>>>(g, t, w, bias, out);
}

void pk_lstm_step(hipStream_t s, const bf16* x, const float* h, const float* c, const bf16* w_ih, const bf16* w_hh,
                  const float* bias, float* h_out, float* c_out, bf16* h_bf16) {
    lstm_step_kernel<<<PRED, 128, 0, s>>>(x, h, c, w_ih, w_hh, bias, h_out, c_out, h_bf16);
}

void pk_linear640(hipStream_t s, const float* h, const bf16* w, const bf16* bias, float* out) {
    linear640_kernel<<<PRED, 128, 0, s>>>(h, w, bias, out);
}

void pk_joint_window(hipStream_t s, const bf16* enc, const float* pred, const int* tT, int rows, bf16* z) {
    joint_window_kernel<<<cdiv(rows * PRED, 256), 256, 0, s>>>(enc, pred, tT, rows, z);
}

void pk_token_duration(hipStream_t s, const bf16* logits, int rows, int* out) {
    token_duration_kernel<<<rows, 256, 0, s>>>(logits, out);
}

void pk_embed(hipStream_t s, const bf16* table, const int* label, bf16* x) {
    embed_kernel<<<cdiv(PRED, 256), 256, 0, s>>>(table, label, x);
}
