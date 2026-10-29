#pragma once
#include <hip/hip_runtime.h>

typedef __bf16 bf16;

enum { ACT_NONE = 0, ACT_GELU = 1 };

// What a matrix product does with each fp32 result before it is stored.
// Rounding follows PyTorch bf16: bias joins the fp32 sum, then each later op
// (activation, residual add) rounds to bf16 again.
struct Epi {
    const bf16* bias = nullptr;
    const bf16* res = nullptr;     // added after rounding; may alias out
    int res_ld = 0;
    int res_mod = 0;               // residual row = row % res_mod when set
    int act = ACT_NONE;
    const int* row_map = nullptr;  // output row per input row, -1 drops it
    bf16* out = nullptr;
    int ldo = 0;
    unsigned long long* argmax = nullptr;  // instead of storing: per-row (bf16 value, lowest index) max
};

struct AttnTile { int q0, q1, k0, k1; };

struct AttnArgs {
    const bf16* q; int q_rs, q_hs;
    const bf16* k; int k_rs, k_hs;
    const bf16* v; int v_rs, v_hs;
    bf16* o; int o_rs, o_hs;
    int group;          // query heads per kv head
    const AttnTile* tiles;
    int n_tiles, n_heads;
    int causal, pos_off;  // causal: key <= query row + pos_off
    float scale;
    // Split-key mode: each tile writes unnormalised fp32 partials ([tile][head][16 rows]) for attention_combine.
    float* part_o = nullptr;
    float* part_ml = nullptr;
};

// 4-bit weights for the speculative draft: per row, groups of 32 along K, nibble = q + 8, fp16 scale.
struct Q4 { const uint4* q; const _Float16* s; };

// C[M][N] = A[M][K] * W[N][K]^T, K a multiple of 32 (8 when M <= 8).
void gemm(hipStream_t s, const bf16* A, int lda, const bf16* W, int M, int N, int K, const Epi& e);
// 3x3 stride-2 pad-1 conv over channels-last [chunk][ih][iw][480] input, as an implicit GEMM.
void conv3x3_480(hipStream_t s, const bf16* in, int chunks, int ih, int iw, const bf16* W, int N, const Epi& e);
// First conv (1 input channel) straight from the log-mel, channels-last [chunk][64][50][480] out, GELU applied.
void conv1(hipStream_t s, const bf16* melT, int fpad, int chunk0, int chunks, const bf16* w, const bf16* b, bf16* out);
// Whisper-style log-mel: out [128][fpad] bf16, frames >= frames are zero.
void log_mel(hipStream_t s, const float* pcm, int n, int frames, int fpad, const float* tables,
             const float* filters, float* scratch, unsigned* gmax, bf16* melT);
void layer_norm(hipStream_t s, const bf16* x, const bf16* w, const bf16* b, bf16* y, int rows, int d);
void rms_norm(hipStream_t s, const bf16* x, int ldx, const bf16* w, bf16* y, int rows, int d);
void qk_norm_rope(hipStream_t s, bf16* qkv, const bf16* qn, const bf16* kn, bf16* kc, bf16* vc,
                  int rows, int pos0, const float* inv_freq);
void attention(hipStream_t s, const AttnArgs& a, int head_dim);
// Merges split-key partials for rows [0, rows) into a.o (head_dim 128).
void attention_combine(hipStream_t s, const AttnArgs& a, int rows, int splits);
void quantize_q4(hipStream_t s, const bf16* w, int N, int K, uint4* q, _Float16* scale);
// One activation row times 4-bit weights, same epilogue rules as gemm.
void gemv_q4(hipStream_t s, const bf16* x, const Q4& w, int N, int K, const Epi& e);
void silu_mul(hipStream_t s, const bf16* gu, bf16* out, int rows, int d);
// Decoder input rows: ids[r] >= 0 takes that embedding row, -1 takes the next audio row.
void embed(hipStream_t s, const int* ids, const bf16* table, const bf16* audio, int audio_start, bf16* out, int rows, int d);
void argmax_ids(hipStream_t s, const unsigned long long* keys, int* ids, int rows);
// fp32 logits of two tokens for each of M normalised rows (d = 2048): out[2 m + j] = x[m] . W[id_j].
void pair_logits(hipStream_t s, const bf16* x, int M, const bf16* W, int id0, int id1, float* out);
