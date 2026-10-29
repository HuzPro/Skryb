// Device kernels for Qwen3-ASR on RDNA 4 (gfx1200, wave32, WMMA bf16).
#include "kernels.h"

#include <algorithm>
#include <cstdint>

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

#define NEG_INF (-__builtin_huge_valf())

namespace {

__device__ __forceinline__ float f(bf16 x) { return (float)x; }
__device__ __forceinline__ bf16 b(float x) { return (bf16)x; }
__device__ __forceinline__ float gelu(float x) { return 0.5f * x * (1.f + erff(x * 0.70710678118654752f)); }
__device__ __forceinline__ unsigned fkey(float v) {
    unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float unkey(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
__device__ __forceinline__ bf16x8 as_bf8(uint4 v) { return __builtin_bit_cast(bf16x8, v); }
__device__ __forceinline__ f32x8 wmma(bf16x8 a, bf16x8 bb, f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, bb, c);
}
__device__ __forceinline__ float lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float dot8(uint4 a, uint4 c, float acc) {
    acc = fmaf(lo(a.x), lo(c.x), acc); acc = fmaf(hi(a.x), hi(c.x), acc);
    acc = fmaf(lo(a.y), lo(c.y), acc); acc = fmaf(hi(a.y), hi(c.y), acc);
    acc = fmaf(lo(a.z), lo(c.z), acc); acc = fmaf(hi(a.z), hi(c.z), acc);
    acc = fmaf(lo(a.w), lo(c.w), acc); acc = fmaf(hi(a.w), hi(c.w), acc);
    return acc;
}
__device__ __forceinline__ float wave_sum(float v) {
    for (int m = 16; m; m >>= 1) v += __shfl_xor(v, m, 32);
    return v;
}
__device__ __forceinline__ float wave_max(float v) {
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor(v, m, 32));
    return v;
}
// Sum over a block of up to 1024 threads; every thread gets the result.
__device__ float block_sum(float v, float* red) {
    v = wave_sum(v);
    int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, waves = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) red[wave] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < waves; i++) t += red[i];
    return t;
}

__device__ __forceinline__ bf16 epi_value(const Epi& e, int row, int col, float v) {
    if (e.bias) v += f(e.bias[col]);
    bf16 r = b(v);
    if (e.act == ACT_GELU) r = b(gelu(f(r)));
    else if (e.act == ACT_SILU) r = b(f(r) / (1.f + expf(-f(r))));
    else if (e.act == ACT_RELU) r = b(fmaxf(f(r), 0.f));
    if (e.scale != 1.f) r = b(f(r) * e.scale);
    if (e.res) {
        int rr = e.res_mod ? row % e.res_mod : row;
        r = b(f(r) + f(e.res[(size_t)rr * e.res_ld + col]));
    }
    return r;
}

// ---- GEMM (WMMA) ----

struct DenseA {
    const bf16* a; int lda;
    __device__ uint4 load(int row, int k, int M) const {
        return row < M ? *(const uint4*)(a + (size_t)row * lda + k) : uint4{0, 0, 0, 0};
    }
};

// Implicit im2col for a 3x3 stride-2 pad-1 conv; k = (ky*3 + kx)*480 + ci.
struct ConvA {
    const bf16* in; int ih, iw, oh, ow;
    __device__ uint4 load(int row, int k, int M) const {
        if (row >= M) return uint4{0, 0, 0, 0};
        int per = oh * ow, chunk = row / per, r = row - chunk * per, oy = r / ow, ox = r - oy * ow;
        int tap = k / 480, ci = k - tap * 480, ky = tap / 3, kx = tap - ky * 3;
        int iy = 2 * oy - 1 + ky, ix = 2 * ox - 1 + kx;
        if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) return uint4{0, 0, 0, 0};
        return *(const uint4*)(in + (((size_t)chunk * ih + iy) * iw + ix) * 480 + ci);
    }
};

// Implicit im2col for a kernel-3 pad-1 1D conv over channels-last input; k = tap * C + ci, C a multiple of 8.
struct Conv1dA {
    const bf16* in; int t_in, C, stride;
    __device__ uint4 load(int row, int k, int M) const {
        const int tap = k / C, ci = k - tap * C, it = stride * row - 1 + tap;
        if (row >= M || tap > 2 || it < 0 || it >= t_in) return uint4{0, 0, 0, 0};
        return *(const uint4*)(in + (size_t)it * C + ci);
    }
};

// Block tile BM x BN, K step BK; 8 waves as 2 (M) x 4 (N), each wave (BM/2) x (BN/4).
// grid.z splits K: each split stores its fp32 partials in ws[split], and gemm_finish sums them and applies
// the epilogue (plain stores: fp32 atomics to VRAM are several times slower here).
// swap_mn puts M on grid.x so neighbouring blocks share one weight tile (long inputs read W once).
template <int BM, int BN, int BK, class AL>
__global__ __launch_bounds__(256) void gemm_kernel(AL A, const bf16* __restrict__ W, int M, int N, int K, int k_split,
                                                   int swap_mn, Epi e, float* ws) {
    constexpr int PAD = BK + 8, WM = BM / 2, WN = BN / 4, TM = WM / 16, TN = WN / 16;
    constexpr int KV = BK / 8, AV = BM * KV / 256, BV = BN * KV / 256;
    __shared__ __attribute__((aligned(16))) bf16 As[2][BM][PAD];
    __shared__ __attribute__((aligned(16))) bf16 Bs[2][BN][PAD];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, wm = wave >> 2, wn = wave & 3;
    const int m0 = (swap_mn ? blockIdx.x : blockIdx.y) * BM, n0 = (swap_mn ? blockIdx.y : blockIdx.x) * BN;
    const int k_begin = blockIdx.z * k_split, k_end = min(K, k_begin + k_split);
    f32x8 acc[TM][TN];
    for (int i = 0; i < TM; i++) for (int j = 0; j < TN; j++) acc[i][j] = f32x8{};
    uint4 ra[AV], rb[BV];
    auto gload = [&](int k0) {
        for (int j = 0; j < AV; j++) {
            int idx = tid + j * 256, r = idx / KV, kv = (idx % KV) * 8;
            ra[j] = A.load(m0 + r, k0 + kv, M);
        }
        for (int j = 0; j < BV; j++) {
            int idx = tid + j * 256, r = idx / KV, kv = (idx % KV) * 8, n = n0 + r;
            rb[j] = n < N ? *(const uint4*)(W + (size_t)n * K + k0 + kv) : uint4{0, 0, 0, 0};
        }
    };
    auto sstore = [&](int buf) {
        for (int j = 0; j < AV; j++) {
            int idx = tid + j * 256;
            *(uint4*)&As[buf][idx / KV][(idx % KV) * 8] = ra[j];
        }
        for (int j = 0; j < BV; j++) {
            int idx = tid + j * 256;
            *(uint4*)&Bs[buf][idx / KV][(idx % KV) * 8] = rb[j];
        }
    };
    gload(k_begin);
    sstore(0);
    __syncthreads();
    const int nk = (k_end - k_begin) / BK;
    for (int kt = 0; kt < nk; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < nk) gload(k_begin + (kt + 1) * BK);
        for (int kk = 0; kk < BK; kk += 16) {
            bf16x8 af[TM], bfr[TN];
            for (int i = 0; i < TM; i++)
                af[i] = as_bf8(*(const uint4*)&As[buf][wm * WM + i * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
            for (int j = 0; j < TN; j++)
                bfr[j] = as_bf8(*(const uint4*)&Bs[buf][wn * WN + j * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++) acc[i][j] = wmma(af[i], bfr[j], acc[i][j]);
        }
        if (kt + 1 < nk) sstore(buf ^ 1);
        __syncthreads();
    }
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++) {
            const int col = n0 + wn * WN + j * 16 + (lane & 15);
            if (col >= N) continue;
            for (int t = 0; t < 8; t++) {
                const int row = m0 + wm * WM + i * 16 + (lane >> 4) * 8 + t;
                if (row >= M) continue;
                if (ws) {
                    ws[((size_t)blockIdx.z * M + row) * N + col] = acc[i][j][t];
                    continue;
                }
                const bf16 r = epi_value(e, row, col, acc[i][j][t]);
                const int orow = e.row_map ? e.row_map[row] : row;
                if (orow >= 0) e.out[(size_t)orow * e.ldo + col] = r;
            }
        }
}

__global__ void gemm_finish_kernel(const float* __restrict__ ws, int splits, int M, int N, Epi e) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    const int row = i / N, col = i - row * N;
    float v = 0.f;
    for (int z = 0; z < splits; z++) v += ws[(size_t)z * M * N + i];
    const bf16 r = epi_value(e, row, col, v);
    const int orow = e.row_map ? e.row_map[row] : row;
    if (orow >= 0) e.out[(size_t)orow * e.ldo + col] = r;
}

// ---- GEMV for M <= 8 rows: weight-bandwidth bound, 2 output columns per wave ----

template <int MR>
__global__ __launch_bounds__(256) void gemv_kernel(const bf16* __restrict__ X, int ldx, int M, const bf16* __restrict__ W,
                                                   int N, int K, Epi e) {
    constexpr int KS = 2048, KV = KS / 8;
    __shared__ uint4 xs[MR * KV];
    __shared__ unsigned long long best[8][MR];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int n0 = blockIdx.x * 16 + wave * 2;
    float acc[2][MR];
    for (int c = 0; c < 2; c++) for (int m = 0; m < MR; m++) acc[c][m] = 0.f;
    for (int k0 = 0; k0 < K; k0 += KS) {
        const int kv_n = min(KS, K - k0) / 8;
        __syncthreads();
        for (int i = threadIdx.x; i < MR * kv_n; i += 256) {
            int m = i / kv_n, kv = i - m * kv_n;
            xs[m * KV + kv] = m < M ? *(const uint4*)(X + (size_t)m * ldx + k0 + kv * 8) : uint4{0, 0, 0, 0};
        }
        __syncthreads();
        const uint4* w0 = (const uint4*)(W + (size_t)min(n0, N - 1) * K + k0);
        const uint4* w1 = (const uint4*)(W + (size_t)min(n0 + 1, N - 1) * K + k0);
        for (int kv = lane; kv < kv_n; kv += 32) {
            uint4 a0 = w0[kv], a1 = w1[kv];
            for (int m = 0; m < MR; m++) {
                uint4 x = xs[m * KV + kv];
                acc[0][m] = dot8(a0, x, acc[0][m]);
                acc[1][m] = dot8(a1, x, acc[1][m]);
            }
        }
    }
    for (int c = 0; c < 2; c++) for (int m = 0; m < MR; m++) acc[c][m] = wave_sum(acc[c][m]);
    if (!e.argmax) {
        if (lane == 0)
            for (int c = 0; c < 2; c++) {
                int n = n0 + c;
                if (n >= N) continue;
                for (int m = 0; m < M && m < MR; m++) {
                    bf16 r = epi_value(e, m, n, acc[c][m]);
                    e.out[(size_t)m * e.ldo + n] = r;
                }
            }
        return;
    }
    if (lane == 0)
        for (int m = 0; m < MR; m++) {
            unsigned long long k = 0;
            for (int c = 0; c < 2; c++) {
                int n = n0 + c;
                if (n >= N) continue;
                unsigned long long key = ((unsigned long long)fkey(f(b(acc[c][m]))) << 32) | (0xffffffffu - (unsigned)n);
                k = key > k ? key : k;
            }
            best[wave][m] = k;
        }
    __syncthreads();
    if (threadIdx.x < MR && threadIdx.x < M) {
        unsigned long long k = 0;
        for (int w = 0; w < 8; w++) k = best[w][threadIdx.x] > k ? best[w][threadIdx.x] : k;
        atomicMax(&e.argmax[threadIdx.x], k);
    }
}

// ---- audio front end ----

// One frame per block: Hann-windowed 400-sample DFT (center, reflect padded), power, mel, log10.
__global__ __launch_bounds__(256) void mel_kernel(const float* __restrict__ pcm, int n, const float* __restrict__ tables,
                                                  const float* __restrict__ filters, float* out, unsigned* gmax) {
    __shared__ float x[400], ct[400], st[400], pw[201], red[8];
    const int t = blockIdx.x, tid = threadIdx.x;
    for (int i = tid; i < 400; i += 256) {
        int s = t * 160 - 200 + i;
        s = s < 0 ? -s : (s >= n ? 2 * (n - 1) - s : s);
        x[i] = pcm[s] * tables[800 + i];
        ct[i] = tables[i];
        st[i] = tables[400 + i];
    }
    __syncthreads();
    if (tid < 201) {
        float re = 0.f, im = 0.f;
        int idx = 0;
        for (int i = 0; i < 400; i++) {
            re = fmaf(x[i], ct[idx], re);
            im = fmaf(x[i], st[idx], im);
            idx += tid;
            if (idx >= 400) idx -= 400;
        }
        pw[tid] = re * re + im * im;
    }
    __syncthreads();
    float v = NEG_INF;
    if (tid < 128) {
        float m = 0.f;
        for (int k = 0; k < 201; k++) m = fmaf(filters[k * 128 + tid], pw[k], m);
        v = log10f(fmaxf(m, 1e-10f));
        out[t * 128 + tid] = v;
    }
    v = wave_max(v);
    if ((tid & 31) == 0) red[tid >> 5] = v;
    __syncthreads();
    if (tid == 0) {
        float mx = red[0];
        for (int i = 1; i < 8; i++) mx = fmaxf(mx, red[i]);
        atomicMax(gmax, fkey(mx));
    }
}

__global__ void mel_finish_kernel(const float* __restrict__ in, int frames, int fpad, const unsigned* gmax, bf16* melT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 128 * fpad) return;
    int j = i / fpad, t = i - j * fpad;
    float floor_ = unkey(*gmax) - 8.f;
    melT[i] = t < frames ? b((fmaxf(in[t * 128 + j], floor_) + 4.f) / 4.f) : b(0.f);
}

__global__ void conv1_kernel(const bf16* __restrict__ melT, int fpad, int chunk0, int chunks, const bf16* __restrict__ w,
                             const bf16* __restrict__ bias, bf16* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunks * 3200 * 480) return;
    int co = idx % 480, p = idx / 480, x = p % 50, y = (p / 50) % 64, c = p / 3200;
    const bf16* col = melT + (size_t)(chunk0 + c) * 100;
    float acc = 0.f;
    for (int ky = 0; ky < 3; ky++) {
        int iy = 2 * y - 1 + ky;
        if (iy < 0 || iy >= 128) continue;
        for (int kx = 0; kx < 3; kx++) {
            int ix = 2 * x - 1 + kx;
            if (ix < 0 || ix >= 100) continue;
            acc = fmaf(f(w[co * 9 + ky * 3 + kx]), f(col[(size_t)iy * fpad + ix]), acc);
        }
    }
    out[idx] = b(gelu(f(b(acc + f(bias[co])))));
}

// ---- norms ----

template <int D>
__global__ __launch_bounds__(256) void layer_norm_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                         const bf16* __restrict__ bb, bf16* y) {
    constexpr int PER = D / 256;
    __shared__ float red[8];
    const bf16* xr = x + (size_t)blockIdx.x * D;
    float v[PER], s = 0.f;
    for (int i = 0; i < PER; i++) { v[i] = f(xr[threadIdx.x * PER + i]); s += v[i]; }
    const float mean = block_sum(s, red) / D;
    float q = 0.f;
    for (int i = 0; i < PER; i++) { float d = v[i] - mean; q += d * d; }
    const float rstd = rsqrtf(block_sum(q, red) / D + 1e-5f);
    for (int i = 0; i < PER; i++) {
        int c = threadIdx.x * PER + i;
        y[(size_t)blockIdx.x * D + c] = b((v[i] - mean) * rstd * f(w[c]) + f(bb[c]));
    }
}

template <int D>
__global__ __launch_bounds__(256) void rms_norm_kernel(const bf16* __restrict__ x, int ldx, const bf16* __restrict__ w, bf16* y) {
    constexpr int PER = D / 256;
    __shared__ float red[8];
    const bf16* xr = x + (size_t)blockIdx.x * ldx;
    float v[PER], s = 0.f;
    for (int i = 0; i < PER; i++) { v[i] = f(xr[threadIdx.x * PER + i]); s += v[i] * v[i]; }
    const float r = rsqrtf(block_sum(s, red) / D + 1e-6f);
    for (int i = 0; i < PER; i++) {
        int c = threadIdx.x * PER + i;
        y[(size_t)blockIdx.x * D + c] = b(f(w[c]) * f(b(v[i] * r)));
    }
}

// grid (rows, 32): slots 0-15 query heads, 16-23 key heads, 24-31 value heads; block 128 = head_dim.
__global__ __launch_bounds__(128) void qk_norm_rope_kernel(bf16* qkv, const bf16* __restrict__ qn, const bf16* __restrict__ kn,
                                                           bf16* kc, bf16* vc, int pos0, const float* __restrict__ inv_freq) {
    __shared__ float red[4], ys[128];
    const int r = blockIdx.x, slot = blockIdx.y, d = threadIdx.x, pos = pos0 + r;
    bf16* row = qkv + (size_t)r * 4096;
    if (slot >= 24) {
        vc[(size_t)pos * 1024 + (slot - 24) * 128 + d] = row[3072 + (slot - 24) * 128 + d];
        return;
    }
    const float x = f(row[slot * 128 + d]);
    const float rr = rsqrtf(block_sum(x * x, red) / 128.f + 1e-6f);
    const float y = f(b(f(slot < 16 ? qn[d] : kn[d]) * f(b(x * rr))));
    ys[d] = y;
    __syncthreads();
    const float partner = d < 64 ? -ys[d + 64] : ys[d - 64];
    const float fr = (float)pos * inv_freq[d & 63];
    const float c = f(b(cosf(fr))), s = f(b(sinf(fr)));
    const bf16 out = b(f(b(y * c)) + f(b(partner * s)));
    if (slot < 16) row[slot * 128 + d] = out;
    else kc[(size_t)pos * 1024 + (slot - 16) * 128 + d] = out;
}

// ---- attention: one wave per (16 query rows, head), online softmax over 16-key tiles ----

template <int D>
__global__ __launch_bounds__(32) void attn_kernel(AttnArgs a) {
    constexpr int KS = D / 16;
    __shared__ __attribute__((aligned(16))) bf16 Ps[16][24];
    __shared__ __attribute__((aligned(16))) bf16 Vs[16][D + 8];
    const int lane = threadIdx.x, h = blockIdx.y, kvh = h / a.group;
    const AttnTile t = a.tiles[blockIdx.x];
    const bf16* Q = a.q + h * a.q_hs;
    const bf16* Kp = a.k + kvh * a.k_hs;
    const bf16* Vp = a.v + kvh * a.v_hs;
    const int half = lane >> 4, col = lane & 15;
    bf16x8 qf[KS];
    {
        const int qrow = t.q0 + col;
        for (int ks = 0; ks < KS; ks++)
            qf[ks] = as_bf8(qrow < t.q1 ? *(const uint4*)(Q + (size_t)qrow * a.q_rs + ks * 16 + half * 8) : uint4{0, 0, 0, 0});
    }
    f32x8 o[KS];
    for (int ks = 0; ks < KS; ks++) o[ks] = f32x8{};
    float m[8], l[8];
    for (int i = 0; i < 8; i++) { m[i] = NEG_INF; l[i] = 0.f; }
    int kend = t.k1;
    if (a.causal) kend = min(kend, t.q1 + a.pos_off);
    for (int kb = t.k0; kb < kend; kb += 16) {
        const int key = kb + col;
        f32x8 s = f32x8{};
        for (int ks = 0; ks < KS; ks++) {
            bf16x8 kf = as_bf8(key < kend ? *(const uint4*)(Kp + (size_t)key * a.k_rs + ks * 16 + half * 8) : uint4{0, 0, 0, 0});
            s = wmma(qf[ks], kf, s);
        }
        for (int j = lane; j < 16 * D / 8; j += 32) {
            int kr = j / (D / 8), dv = (j - kr * (D / 8)) * 8, kk = kb + kr;
            *(uint4*)&Vs[kr][dv] = kk < kend ? *(const uint4*)(Vp + (size_t)kk * a.v_rs + dv) : uint4{0, 0, 0, 0};
        }
        for (int i = 0; i < 8; i++) {
            const int qr = t.q0 + half * 8 + i;
            const bool ok = key < kend && (!a.causal || key <= qr + a.pos_off);
            const float v = ok ? s[i] * a.scale : NEG_INF;
            float mx = v;
            for (int x = 1; x < 16; x <<= 1) mx = fmaxf(mx, __shfl_xor(mx, x, 32));
            const float mn = fmaxf(m[i], mx);
            const float p = mn == NEG_INF ? 0.f : expf(v - mn);
            const float alpha = mn == NEG_INF ? 1.f : expf(m[i] - mn);
            float ps = p;
            for (int x = 1; x < 16; x <<= 1) ps += __shfl_xor(ps, x, 32);
            l[i] = l[i] * alpha + ps;
            m[i] = mn;
            for (int ks = 0; ks < KS; ks++) o[ks][i] *= alpha;
            Ps[half * 8 + i][col] = b(p);
        }
        __syncthreads();
        const bf16x8 pf = as_bf8(*(const uint4*)&Ps[col][half * 8]);
        for (int nt = 0; nt < KS; nt++) {
            bf16x8 vf;
            for (int j = 0; j < 8; j++) vf[j] = Vs[half * 8 + j][nt * 16 + col];
            o[nt] = wmma(pf, vf, o[nt]);
        }
        __syncthreads();
    }
    if (a.part_o) {
        const size_t base = ((size_t)blockIdx.x * a.n_heads + h) * 16;
        for (int i = 0; i < 8; i++) {
            const size_t r = base + half * 8 + i;
            for (int nt = 0; nt < KS; nt++) a.part_o[r * D + nt * 16 + col] = o[nt][i];
            if (col == 0) { a.part_ml[r * 2] = m[i]; a.part_ml[r * 2 + 1] = l[i]; }
        }
        return;
    }
    for (int i = 0; i < 8; i++) {
        const int row = t.q0 + half * 8 + i;
        if (row >= t.q1) continue;
        const float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
        for (int nt = 0; nt < KS; nt++) a.o[(size_t)row * a.o_rs + h * a.o_hs + nt * 16 + col] = b(o[nt][i] * inv);
    }
}

// grid (rows, heads), block D. A split holding only masked keys has m = -inf and adds exactly nothing,
// so the result does not depend on how many trailing splits a call happens to include.
template <int D>
__global__ __launch_bounds__(D) void attn_combine_kernel(AttnArgs a, int splits) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float mx = NEG_INF;
    for (int sp = 0; sp < splits; sp++) mx = fmaxf(mx, a.part_ml[(((size_t)sp * a.n_heads + h) * 16 + r) * 2]);
    float lsum = 0.f, osum = 0.f;
    for (int sp = 0; sp < splits; sp++) {
        const size_t i = ((size_t)sp * a.n_heads + h) * 16 + r;
        const float m = a.part_ml[i * 2];
        if (m == NEG_INF) continue;
        const float wgt = expf(m - mx);
        lsum += wgt * a.part_ml[i * 2 + 1];
        osum += wgt * a.part_o[i * D + d];
    }
    const float inv = lsum > 0.f ? 1.f / lsum : 0.f;
    a.o[(size_t)r * a.o_rs + h * a.o_hs + d] = b(osum * inv);
}

// One thread per (row, group of 32).
__global__ void quantize_q4_kernel(const bf16* __restrict__ w, int N, int K, uint4* q, _Float16* scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, groups = K / 32;
    if (i >= N * groups) return;
    const bf16* src = w + (size_t)i * 32;
    float amax = 0.f;
    for (int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(f(src[j])));
    const _Float16 sh = (_Float16)(amax > 0.f ? amax / 7.f : 1.f);
    const float inv = 1.f / (float)sh;
    unsigned words[4] = {0, 0, 0, 0};
    for (int j = 0; j < 32; j++) {
        int v = (int)rintf(f(src[j]) * inv);
        v = max(-8, min(7, v)) + 8;
        words[j >> 3] |= (unsigned)v << ((j & 7) * 4);
    }
    q[i] = uint4{words[0], words[1], words[2], words[3]};
    scale[i] = sh;
}

__device__ __forceinline__ float dot32_q4(uint4 qv, const uint4* x) {
    const unsigned qw[4] = {qv.x, qv.y, qv.z, qv.w};
    float acc = 0.f;
    for (int j = 0; j < 4; j++) {
        const uint4 xv = x[j];
        const unsigned xs[4] = {xv.x, xv.y, xv.z, xv.w};
        for (int p = 0; p < 4; p++) {
            acc = fmaf((float)((int)((qw[j] >> (p * 8)) & 15) - 8), lo(xs[p]), acc);
            acc = fmaf((float)((int)((qw[j] >> (p * 8 + 4)) & 15) - 8), hi(xs[p]), acc);
        }
    }
    return acc;
}

// 256 threads, 2 output columns per wave, the activation row staged in LDS.
__global__ __launch_bounds__(256) void gemv_q4_kernel(const bf16* __restrict__ X, Q4 w, int N, int K, Epi e) {
    __shared__ uint4 xs[6144 / 8];
    __shared__ unsigned long long best[8];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, n0 = blockIdx.x * 16 + wave * 2, groups = K / 32;
    for (int i = threadIdx.x; i < K / 8; i += 256) xs[i] = ((const uint4*)X)[i];
    __syncthreads();
    float acc[2] = {0.f, 0.f};
    for (int c = 0; c < 2; c++) {
        const int n = min(n0 + c, N - 1);
        const uint4* qr = w.q + (size_t)n * groups;
        const _Float16* sr = w.s + (size_t)n * groups;
        for (int g = lane; g < groups; g += 32) acc[c] = fmaf((float)sr[g], dot32_q4(qr[g], xs + g * 4), acc[c]);
        acc[c] = wave_sum(acc[c]);
    }
    if (!e.argmax) {
        if (lane == 0)
            for (int c = 0; c < 2; c++)
                if (n0 + c < N) e.out[n0 + c] = epi_value(e, 0, n0 + c, acc[c]);
        return;
    }
    if (lane == 0) {
        unsigned long long k = 0;
        for (int c = 0; c < 2; c++) {
            if (n0 + c >= N) continue;
            unsigned long long key = ((unsigned long long)fkey(f(b(acc[c]))) << 32) | (0xffffffffu - (unsigned)(n0 + c));
            k = key > k ? key : k;
        }
        best[wave] = k;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long k = 0;
        for (int i = 0; i < 8; i++) k = best[i] > k ? best[i] : k;
        atomicMax(e.argmax, k);
    }
}

__global__ void silu_mul_kernel(const bf16* __restrict__ gu, bf16* out, int rows, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * d) return;
    int r = i / d, c = i - r * d;
    float g = f(gu[(size_t)r * 2 * d + c]);
    out[i] = b(f(b(g / (1.f + expf(-g)))) * f(gu[(size_t)r * 2 * d + d + c]));
}

__global__ void embed_kernel(const int* __restrict__ ids, const bf16* __restrict__ table, const bf16* __restrict__ audio,
                             int audio_start, bf16* out, int d) {
    const int r = blockIdx.x, id = ids[r];
    const uint4* src = (const uint4*)(id >= 0 ? table + (size_t)id * d : audio + (size_t)(r - audio_start) * d);
    uint4* dst = (uint4*)(out + (size_t)r * d);
    for (int i = threadIdx.x; i < d / 8; i += blockDim.x) dst[i] = src[i];
}

__global__ void argmax_ids_kernel(const unsigned long long* keys, int* ids, int rows) {
    int i = threadIdx.x;
    if (i < rows) ids[i] = (int)(0xffffffffu - (unsigned)(keys[i] & 0xffffffffu));
}

__global__ __launch_bounds__(256) void pair_logits_kernel(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                                          int id0, int id1, float* out) {
    __shared__ float red[8];
    const int m = blockIdx.x, j = blockIdx.y;
    const bf16* xr = x + (size_t)m * 2048;
    const bf16* wr = W + (size_t)(j ? id1 : id0) * 2048;
    float acc = 0.f;
    for (int i = threadIdx.x; i < 2048; i += 256) acc = fmaf(f(xr[i]), f(wr[i]), acc);
    acc = block_sum(acc, red);
    if (threadIdx.x == 0) out[m * 2 + j] = acc;
}

int cdiv(int a, int c) { return (a + c - 1) / c; }

}  // namespace

namespace {

float* split_workspace(size_t floats) {
    static float* ws = nullptr;
    static size_t cap = 0;
    if (floats > cap) {
        if (ws) (void)hipFree(ws);
        cap = std::max(floats, (size_t)4 * 512 * 4096);
        if (hipMalloc(&ws, cap * 4) != hipSuccess) { ws = nullptr; cap = 0; }
    }
    return ws;
}

// Long inputs: 128x128 tiles with M on grid.x. Short ones (dictation prefill): 64-wide column tiles so
// each weight is read once, and K split until at least 128 blocks keep all 32 CUs busy.
template <class AL>
void launch_gemm(hipStream_t s, const AL& a, const bf16* W, int M, int N, int K, const Epi& e) {
    if (M >= 512 || K % 64) {
        gemm_kernel<128, 128, 32><<<dim3(cdiv(M, 128), cdiv(N, 128), 1), 256, 0, s>>>(a, W, M, N, K, K, 1, e, nullptr);
        return;
    }
    const int bm = M > 64 ? 128 : 64, tiles = cdiv(M, bm) * cdiv(N, 64);
    int splits = 1;
    while (tiles * splits < 128 && K % (splits * 2 * 64) == 0 && K / (splits * 2) >= 512) splits *= 2;
    float* ws = splits > 1 ? split_workspace((size_t)splits * M * N) : nullptr;
    if (splits > 1 && !ws) splits = 1;
    const dim3 g(cdiv(N, 64), cdiv(M, bm), splits);
    if (bm == 128) gemm_kernel<128, 64, 64><<<g, 256, 0, s>>>(a, W, M, N, K, K / splits, 0, e, ws);
    else gemm_kernel<64, 64, 64><<<g, 256, 0, s>>>(a, W, M, N, K, K / splits, 0, e, ws);
    if (ws) gemm_finish_kernel<<<cdiv(M * N, 256), 256, 0, s>>>(ws, splits, M, N, e);
}

}  // namespace

void gemm(hipStream_t s, const bf16* A, int lda, const bf16* W, int M, int N, int K, const Epi& e) {
    if (M <= 8) {
        dim3 g(cdiv(N, 16));
        if (M == 1) gemv_kernel<1><<<g, 256, 0, s>>>(A, lda, M, W, N, K, e);
        else if (M == 2) gemv_kernel<2><<<g, 256, 0, s>>>(A, lda, M, W, N, K, e);
        else if (M <= 4) gemv_kernel<4><<<g, 256, 0, s>>>(A, lda, M, W, N, K, e);
        else gemv_kernel<8><<<g, 256, 0, s>>>(A, lda, M, W, N, K, e);
        return;
    }
    launch_gemm(s, DenseA{A, lda}, W, M, N, K, e);
}

void conv3x3_480(hipStream_t s, const bf16* in, int chunks, int ih, int iw, const bf16* W, int N, const Epi& e) {
    const int oh = (ih + 1) / 2, ow = (iw + 1) / 2;
    launch_gemm(s, ConvA{in, ih, iw, oh, ow}, W, chunks * oh * ow, N, 4320, e);
}

void conv1d_k3(hipStream_t s, const bf16* in, int t_in, int C, int stride, const bf16* W, int M, int N, int K, const Epi& e) {
    launch_gemm(s, Conv1dA{in, t_in, C, stride}, W, M, N, K, e);
}

void conv1(hipStream_t s, const bf16* melT, int fpad, int chunk0, int chunks, const bf16* w, const bf16* bias, bf16* out) {
    int n = chunks * 3200 * 480;
    conv1_kernel<<<cdiv(n, 256), 256, 0, s>>>(melT, fpad, chunk0, chunks, w, bias, out);
}

void log_mel(hipStream_t s, const float* pcm, int n, int frames, int fpad, const float* tables, const float* filters,
             float* scratch, unsigned* gmax, bf16* melT) {
    hipMemsetAsync(gmax, 0, 4, s);
    mel_kernel<<<frames, 256, 0, s>>>(pcm, n, tables, filters, scratch, gmax);
    mel_finish_kernel<<<cdiv(128 * fpad, 256), 256, 0, s>>>(scratch, frames, fpad, gmax, melT);
}

void layer_norm(hipStream_t s, const bf16* x, const bf16* w, const bf16* bb, bf16* y, int rows, int d) {
    if (d == 1024) layer_norm_kernel<1024><<<rows, 256, 0, s>>>(x, w, bb, y);
    else if (d == 1280) layer_norm_kernel<1280><<<rows, 256, 0, s>>>(x, w, bb, y);
}

void rms_norm(hipStream_t s, const bf16* x, int ldx, const bf16* w, bf16* y, int rows, int d) {
    if (d == 2048) rms_norm_kernel<2048><<<rows, 256, 0, s>>>(x, ldx, w, y);
}

void qk_norm_rope(hipStream_t s, bf16* qkv, const bf16* qn, const bf16* kn, bf16* kc, bf16* vc, int rows, int pos0,
                  const float* inv_freq) {
    qk_norm_rope_kernel<<<dim3(rows, 32), 128, 0, s>>>(qkv, qn, kn, kc, vc, pos0, inv_freq);
}

void attention(hipStream_t s, const AttnArgs& a, int head_dim) {
    dim3 g(a.n_tiles, a.n_heads);
    if (head_dim == 64) attn_kernel<64><<<g, 32, 0, s>>>(a);
    else attn_kernel<128><<<g, 32, 0, s>>>(a);
}

void attention_combine(hipStream_t s, const AttnArgs& a, int rows, int splits, int head_dim) {
    if (head_dim == 64) attn_combine_kernel<64><<<dim3(rows, a.n_heads), 64, 0, s>>>(a, splits);
    else attn_combine_kernel<128><<<dim3(rows, a.n_heads), 128, 0, s>>>(a, splits);
}

void quantize_q4(hipStream_t s, const bf16* w, int N, int K, uint4* q, _Float16* scale) {
    quantize_q4_kernel<<<cdiv(N * (K / 32), 256), 256, 0, s>>>(w, N, K, q, scale);
}

void gemv_q4(hipStream_t s, const bf16* x, const Q4& w, int N, int K, const Epi& e) {
    gemv_q4_kernel<<<cdiv(N, 16), 256, 0, s>>>(x, w, N, K, e);
}

void silu_mul(hipStream_t s, const bf16* gu, bf16* out, int rows, int d) {
    silu_mul_kernel<<<cdiv(rows * d, 256), 256, 0, s>>>(gu, out, rows, d);
}

void embed(hipStream_t s, const int* ids, const bf16* table, const bf16* audio, int audio_start, bf16* out, int rows, int d) {
    embed_kernel<<<rows, 256, 0, s>>>(ids, table, audio, audio_start, out, d);
}

void pair_logits(hipStream_t s, const bf16* x, int M, const bf16* W, int id0, int id1, float* out) {
    pair_logits_kernel<<<dim3(M, 2), 256, 0, s>>>(x, W, id0, id1, out);
}

void argmax_ids(hipStream_t s, const unsigned long long* keys, int* ids, int rows) {
    argmax_ids_kernel<<<1, 32, 0, s>>>(keys, ids, rows);
}
