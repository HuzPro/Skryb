#pragma once
// Native Qwen3-ASR-1.7B engine for one AMD GPU. Not thread-safe: one call at a time per engine.
#ifdef __cplusplus
extern "C" {
#endif

#ifdef QASR_BUILD
#define QASR_API __declspec(dllexport)
#else
#define QASR_API __declspec(dllimport)
#endif

typedef struct QasrEngine QasrEngine;

// Loads a qasr.bin made by convert.py. Returns NULL and fills err on failure.
QASR_API QasrEngine* qasr_load(const char* path, char* err, int err_len);
// 16 kHz mono float samples in [-1, 1]. Writes UTF-8 text; returns its length, or -1 (text holds the error).
QASR_API int qasr_transcribe(QasrEngine* engine, const float* pcm, int samples, char* text, int text_len);
// JSON timings of the last load or transcribe call.
QASR_API const char* qasr_stats(QasrEngine* engine);
QASR_API void qasr_free(QasrEngine* engine);

#ifdef __cplusplus
}
#endif
