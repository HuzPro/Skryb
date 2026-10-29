#pragma once
// Kernels only Whisper needs; the GEMM, norms and attention come from ../engine/kernels.h.
#include "../engine/kernels.h"

// Whisper's log-mel over frames of a 30 s signal (centred periodic-Hann 400-sample DFT, reflect padded):
// out [frames][n_mels] bf16, channels last. tables: cos[400], sin[400], hann[400]; filters [201][n_mels].
void wh_log_mel(hipStream_t s, const float* pcm, int n, int frames, int n_mels, const float* tables,
                const float* filters, float* scratch, unsigned* gmax, bf16* out);
// out[r] = embed[ids[r]] + pos[pos0 + r], rounded to bf16 like the model's own add.
void wh_embed_pos(hipStream_t s, const int* ids, const bf16* embed, const bf16* pos, int pos0, bf16* out, int rows, int d);
// Conv weight [N][C][3] -> [N][K] with k = tap * C + ci, zero for k >= 3 * C.
void wh_conv_weight(hipStream_t s, const bf16* src, int N, int C, int K, bf16* dst);
// ids[r] = the lowest index of the largest of the n bf16 logits in row r (rows ld apart).
void wh_argmax(hipStream_t s, const bf16* logits, int n, int ld, int rows, int* ids);

