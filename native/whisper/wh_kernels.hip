// Device kernels for the native Whisper engine on RDNA 4 (gfx1200, wave32).
#include "wh_kernels.h"

#define NEG_INF (-__builtin_huge_valf())

namespace {

__device__ __forceinline__ float f(bf16 x) { return (float)x; }
__device__ __forceinline__ bf16 b(float x) { return (bf16)x; }
__device__ __forceinline__ unsigned fkey(float v) {
    unsigned u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float unkey(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
__device__ __forceinline__ float wave_max(float v) {
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor(v, m, 32));
    return v;
}

// One frame per block, as WhisperFeatureExtractor: power of the DFT, mel, log10 of at least 1e-10.
__global__ __launch_bounds__(256) void mel_kernel(const float* __restrict__ pcm, int n, int nm,
                                                  const float* __restrict__ tables, const float* __restrict__ filters,
                                                  float* out, unsigned* gmax) {
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
    if (tid < nm) {
        float m = 0.f;
        for (int k = 0; k < 201; k++) m = fmaf(filters[k * nm + tid], pw[k], m);
        v = log10f(fmaxf(m, 1e-10f));
        out[t * nm + tid] = v;
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

// Clamp to 8 below the clip's peak, then (x + 4) / 4.
__global__ void mel_finish_kernel(const float* __restrict__ in, int count, const unsigned* gmax, bf16* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float floor_ = unkey(*gmax) - 8.f;
    out[i] = b((fmaxf(in[i], floor_) + 4.f) / 4.f);
}

__global__ void embed_pos_kernel(const int* __restrict__ ids, const bf16* __restrict__ embed, const bf16* __restrict__ pos,
                                 int pos0, bf16* out, int d) {
    const int r = blockIdx.x;
    const bf16* e = embed + (size_t)ids[r] * d;
    const bf16* p = pos + (size_t)(pos0 + r) * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) out[(size_t)r * d + c] = b(f(e[c]) + f(p[c]));
}

__global__ void conv_weight_kernel(const bf16* __restrict__ src, int N, int C, int K, bf16* dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * K) return;
    const int n = i / K, k = i - n * K, tap = k / C, ci = k - tap * C;
    dst[i] = tap < 3 ? src[((size_t)n * C + ci) * 3 + tap] : b(0.f);
}

__global__ __launch_bounds__(1024) void argmax_kernel(const bf16* __restrict__ logits, int n, int* ids) {
    __shared__ unsigned long long best[32];
    unsigned long long k = 0;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const unsigned long long key = ((unsigned long long)fkey(f(logits[i])) << 32) | (0xffffffffu - (unsigned)i);
        k = key > k ? key : k;
    }
    for (int m = 16; m; m >>= 1) {
        const unsigned long long o = __shfl_xor(k, m, 32);
        k = o > k ? o : k;
    }
    if ((threadIdx.x & 31) == 0) best[threadIdx.x >> 5] = k;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; w++) k = best[w] > k ? best[w] : k;
        ids[0] = (int)(0xffffffffu - (unsigned)(k & 0xffffffffu));
    }
}

__global__ void step_begin_kernel(const int* __restrict__ pos, const int* __restrict__ ids, const bf16* __restrict__ embed,
                                  const bf16* __restrict__ pos_table, bf16* out, AttnTile* tile, int d) {
    const int p = *pos;
    const bf16* e = embed + (size_t)ids[p] * d;
    const bf16* q = pos_table + (size_t)p * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) out[c] = b(f(e[c]) + f(q[c]));
    if (threadIdx.x == 0) *tile = AttnTile{0, 1, 0, p + 1};
}

__global__ void store_kv_kernel(const int* __restrict__ pos, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                bf16* kc, bf16* vc, int d) {
    const size_t row = (size_t)*pos * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        kc[row + c] = k[c];
        vc[row + c] = v[c];
    }
}

// The argmax of argmax_kernel, written where the next step reads it.
__global__ __launch_bounds__(1024) void argmax_step_kernel(const bf16* __restrict__ logits, int n, int* ids, int* pos, int* last) {
    __shared__ unsigned long long best[32];
    unsigned long long k = 0;
    for (int i = threadIdx.x; i < n; i += 1024) {
        const unsigned long long key = ((unsigned long long)fkey(f(logits[i])) << 32) | (0xffffffffu - (unsigned)i);
        k = key > k ? key : k;
    }
    for (int m = 16; m; m >>= 1) {
        const unsigned long long o = __shfl_xor(k, m, 32);
        k = o > k ? o : k;
    }
    if ((threadIdx.x & 31) == 0) best[threadIdx.x >> 5] = k;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 32; w++) k = best[w] > k ? best[w] : k;
        const int id = (int)(0xffffffffu - (unsigned)(k & 0xffffffffu)), p = *pos;
        ids[p + 1] = id;
        *last = id;
        *pos = p + 1;
    }
}

int cdiv(int a, int c) { return (a + c - 1) / c; }

}  // namespace

void wh_log_mel(hipStream_t s, const float* pcm, int n, int frames, int n_mels, const float* tables,
                const float* filters, float* scratch, unsigned* gmax, bf16* out) {
    hipMemsetAsync(gmax, 0, 4, s);
    mel_kernel<<<frames, 256, 0, s>>>(pcm, n, n_mels, tables, filters, scratch, gmax);
    mel_finish_kernel<<<cdiv(frames * n_mels, 256), 256, 0, s>>>(scratch, frames * n_mels, gmax, out);
}

void wh_embed_pos(hipStream_t s, const int* ids, const bf16* embed, const bf16* pos, int pos0, bf16* out, int rows, int d) {
    embed_pos_kernel<<<rows, 256, 0, s>>>(ids, embed, pos, pos0, out, d);
}

void wh_conv_weight(hipStream_t s, const bf16* src, int N, int C, int K, bf16* dst) {
    conv_weight_kernel<<<cdiv(N * K, 256), 256, 0, s>>>(src, N, C, K, dst);
}

void wh_argmax(hipStream_t s, const bf16* logits, int n, int* ids) {
    argmax_kernel<<<1, 1024, 0, s>>>(logits, n, ids);
}

void wh_step_begin(hipStream_t s, const int* pos, const int* ids, const bf16* embed, const bf16* pos_table, bf16* out,
                   AttnTile* tile, int d) {
    step_begin_kernel<<<1, 256, 0, s>>>(pos, ids, embed, pos_table, out, tile, d);
}

void wh_store_kv(hipStream_t s, const int* pos, const bf16* k, const bf16* v, bf16* k_cache, bf16* v_cache, int d) {
    store_kv_kernel<<<1, 256, 0, s>>>(pos, k, v, k_cache, v_cache, d);
}

void wh_argmax_step(hipStream_t s, const bf16* logits, int n, int* ids, int* pos, int* last) {
    argmax_step_kernel<<<1, 1024, 0, s>>>(logits, n, ids, pos, last);
}
