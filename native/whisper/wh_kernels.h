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
// ids[0] = the lowest index of the largest of n bf16 logits.
void wh_argmax(hipStream_t s, const bf16* logits, int n, int* ids);

// Graph-replayable decode step pieces: the position lives on the device in *pos.
// Row 0 of out = embed[ids[p]] + pos_table[p], and *tile = {0, 1, 0, p + 1} (that row's causal keys).
void wh_step_begin(hipStream_t s, const int* pos, const int* ids, const bf16* embed, const bf16* pos_table, bf16* out,
                   AttnTile* tile, int d);
// k_cache[p] = k, v_cache[p] = v (one row each).
void wh_store_kv(hipStream_t s, const int* pos, const bf16* k, const bf16* v, bf16* k_cache, bf16* v_cache, int d);
// ids[p + 1] = *last = argmax of the logits, then *pos = p + 1.
void wh_argmax_step(hipStream_t s, const bf16* logits, int n, int* ids, int* pos, int* last);
