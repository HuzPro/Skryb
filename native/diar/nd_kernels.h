#pragma once
#include "../engine/kernels.h"

void nd_to_bf16(hipStream_t s, const float* src, size_t n, bf16* dst);
void nd_to_f32(hipStream_t s, const bf16* src, size_t n, float* dst);
// Conv weight [N][C][3] fp32 -> [N][K] bf16 with k = tap * C + ci, zero for k >= 3 * C.
void nd_conv_weight(hipStream_t s, const float* src, int N, int C, int K, bf16* dst);
// Rotary position embedding on q and k ([rows][heads * 64], ld apart), positions 0..rows-1, in place.
void nd_rope(hipStream_t s, bf16* q, bf16* k, int ld, int rows, int heads, const float* inv_freq);
