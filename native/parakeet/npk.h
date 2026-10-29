#pragma once
// Native Parakeet TDT 0.6B engine (FastConformer + TDT) for one AMD GPU (gfx1200). Serves Parakeet Ultra and
// the Underdog Bark family. Not thread-safe: one call at a time per engine.
#ifdef __cplusplus
extern "C" {
#endif

#ifdef NPK_BUILD
#define NPK_API __declspec(dllexport)
#else
#define NPK_API __declspec(dllimport)
#endif

typedef struct NpkEngine NpkEngine;

// Loads a parakeet.bin made by skryb/transcription/parakeet_pack.py. Returns NULL and fills err on failure.
NPK_API NpkEngine* npk_load(const char* path, char* err, int err_len);
// 16 kHz mono float samples in [-1, 1]. Writes UTF-8 text; returns its length, or -1 (text holds the error).
NPK_API int npk_transcribe(NpkEngine* engine, const float* pcm, int samples, char* text, int text_len);
// JSON timings of the last load or transcribe call.
NPK_API const char* npk_stats(NpkEngine* engine);
NPK_API void npk_free(NpkEngine* engine);

#ifdef __cplusplus
}
#endif
