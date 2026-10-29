#pragma once
// Native Nemotron 3 Diarization engine (the encoder, upsampler and speaker classifier of one streaming step)
// for one AMD GPU (gfx1200). The feature extractor and the speaker cache stay in Python. Not thread-safe.
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifdef ND_BUILD
#define ND_API __declspec(dllexport)
#else
#define ND_API __declspec(dllimport)
#endif

typedef struct NdEngine NdEngine;

// Loads the fp32 model.safetensors as it is on disk: names[i] is a tensor's Hugging Face name and offsets[i] the
// file offset of its first byte, every one a multiple of 16 apart. Weights are kept in bf16. Returns NULL and
// fills err on failure.
ND_API NdEngine* nd_load(const char* path, const char* const* names, const uint64_t* offsets, int count, int layers,
                         char* err, int err_len);
// One step: embeds is the encoder input [rows][512] (speaker cache, FIFO, chunk and look-ahead), rows from
// valid on are padding no frame attends to. Writes the speaker logits [rows * 8][8]; returns 0, or -1 with the
// error in err.
ND_API int nd_forward(NdEngine* engine, const float* embeds, int rows, int valid, float* logits, char* err, int err_len);
// JSON timings of the last load or forward call.
ND_API const char* nd_stats(NdEngine* engine);
ND_API void nd_free(NdEngine* engine);

#ifdef __cplusplus
}
#endif
