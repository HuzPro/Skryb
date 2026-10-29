// Device kernels only the diarizer needs; the GEMM, norms and attention come from ../engine/kernels.hip.
#include "nd_kernels.h"

namespace {

__device__ __forceinline__ float f(bf16 x) { return (float)x; }
__device__ __forceinline__ bf16 b(float x) { return (bf16)x; }

__global__ void to_bf16_kernel(const float* __restrict__ src, size_t n, bf16* dst) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = b(src[i]);
}

__global__ void to_f32_kernel(const bf16* __restrict__ src, size_t n, float* dst) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f(src[i]);
}

__global__ void conv_weight_kernel(const float* __restrict__ src, int N, int C, int K, bf16* dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * K) return;
    const int n = i / K, k = i - n * K, tap = k / C, ci = k - tap * C;
    dst[i] = tap < 3 ? b(src[((size_t)n * C + ci) * 3 + tap]) : b(0.f);
}

// grid (rows, heads), block 64: transformers' apply_rotary_pos_emb (rotate_half) on q and k, in fp32.
__global__ __launch_bounds__(64) void rope_kernel(bf16* q, bf16* k, int ld, const float* __restrict__ inv_freq) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x, half = d < 32 ? d : d - 32;
    const float a = (float)r * inv_freq[half], c = cosf(a), s = sinf(a);
    bf16* rows[2] = {q + (size_t)r * ld + h * 64, k + (size_t)r * ld + h * 64};
    float x[2], partner[2];
    for (int j = 0; j < 2; j++) {
        x[j] = f(rows[j][d]);
        partner[j] = d < 32 ? -f(rows[j][d + 32]) : f(rows[j][d - 32]);
    }
    __syncthreads();
    for (int j = 0; j < 2; j++) rows[j][d] = b(x[j] * c + partner[j] * s);
}

int cdiv(size_t a, int c) { return (int)((a + c - 1) / c); }

}  // namespace

void nd_to_bf16(hipStream_t s, const float* src, size_t n, bf16* dst) {
    to_bf16_kernel<<<cdiv(n, 256), 256, 0, s>>>(src, n, dst);
}

void nd_to_f32(hipStream_t s, const bf16* src, size_t n, float* dst) {
    to_f32_kernel<<<cdiv(n, 256), 256, 0, s>>>(src, n, dst);
}

void nd_conv_weight(hipStream_t s, const float* src, int N, int C, int K, bf16* dst) {
    conv_weight_kernel<<<cdiv((size_t)N * K, 256), 256, 0, s>>>(src, N, C, K, dst);
}

void nd_rope(hipStream_t s, bf16* q, bf16* k, int ld, int rows, int heads, const float* inv_freq) {
    rope_kernel<<<dim3(rows, heads), 64, 0, s>>>(q, k, ld, inv_freq);
}
