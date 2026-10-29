// Kernels particular to Parakeet TDT (FastConformer + TDT decoder) on gfx1200; GEMMs and layer norm come
// from ../engine/kernels.hip.
#pragma once
#include "../engine/kernels.h"

// NeMo's featurizer: pre-emphasis 0.97, 512-point STFT (400-sample symmetric Hann centred in it, hop 160,
// reflect padding), power, 128 mel bands, log(x + 2^-24). out [frames][128] fp32.
void pk_log_mel(hipStream_t s, const float* pcm, int n, int frames, const float* window512, const float* cos_sin,
                const float* fb, float* out);
// Per-band normalisation over time (unbiased std, + 1e-5), written as bf16 [frames][128].
void pk_normalize(hipStream_t s, const float* mel, int frames, bf16* out);
// First subsampling conv: 1 -> 256 channels, 3x3 stride 2 pad 1, ReLU. in [t][128], out [t'][64][256].
void pk_conv0(hipStream_t s, const bf16* in, int t, const bf16* w, const bf16* bias, bf16* out);
// Depthwise 3x3 stride 2 pad 1 over [t][f][256] channels-last; out [t'][f'][256].
void pk_dwconv(hipStream_t s, const bf16* in, int t, int f, const bf16* w, const bf16* bias, bf16* out);
// Relative-position attention, 8 heads of 128: scores = ((q + u) k + rel_shift((q + v) p)) / sqrt(128).
// qkv [t][3072] (q | k | v), p [2t - 1][1024] (row r is relative position t - 1 - r), out [t][1024].
void pk_relpos_attention(hipStream_t s, const bf16* qkv, const bf16* p, const bf16* u, const bf16* v, int t, bf16* out);
// GLU over [t][2048], depthwise conv (kernel 9, pad 4) with batch norm folded in, SiLU: out [t][1024].
void pk_glu_dwconv_silu(hipStream_t s, const bf16* g, int t, const float* w, const float* bias, bf16* out);
// One LSTM layer step (PyTorch gate order i, f, g, o) for 640 units: fp32 state, bf16 weights.
void pk_lstm_step(hipStream_t s, const bf16* x, const float* h, const float* c, const bf16* w_ih, const bf16* w_hh,
                  const float* bias, float* h_out, float* c_out, bf16* h_bf16);
// pred = W h + b (640 x 640) in fp32.
void pk_linear640(hipStream_t s, const float* h, const bf16* w, const bf16* bias, float* out);
// z[r][j] = bf16(relu(enc[t0 + r][j] + pred[j])) for r < rows, zero past the last frame; t0 = tT[0], frames = tT[1]
// (read on the device, so one captured graph serves every window).
void pk_joint_window(hipStream_t s, const bf16* enc, const float* pred, const int* tT, int rows, bf16* z);
// Per row of logits [rows][8198]: the token argmax over the first 8193 and the duration argmax over the last 5.
void pk_token_duration(hipStream_t s, const bf16* logits, int rows, int* out);
// Embedding row of `label` (device int) into x.
void pk_embed(hipStream_t s, const bf16* table, const int* label, bf16* x);
