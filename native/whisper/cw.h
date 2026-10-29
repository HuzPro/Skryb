#pragma once
// Native CrisperWhisper 2.0 engine (Whisper encoder-decoder, Large and Turbo) for one AMD GPU (gfx1200).
// Not thread-safe: one call at a time per engine.
#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#ifdef CW_BUILD
#define CW_API __declspec(dllexport)
#else
#define CW_API __declspec(dllimport)
#endif

typedef struct CwEngine CwEngine;

// The model's shape, from its config.json.
typedef struct CwConfig {
    int n_mels, encoder_layers, decoder_layers, vocab, eos;
} CwConfig;

// Loads a model.safetensors as it is on disk: names[i] is a tensor's Hugging Face name and offsets[i] the
// file offset of its first byte. Every tensor must sit at the same offset modulo 16; the file from the
// lowest offset on is copied to VRAM in one pass. suppress lists the token ids greedy decoding never picks.
// Returns NULL and fills err on failure.
CW_API CwEngine* cw_load(const char* path, const char* const* names, const uint64_t* offsets, int count,
                         const CwConfig* config, const int* suppress, int n_suppress, char* err, int err_len);
// At most 30 s of 16 kHz mono float samples in [-1, 1], padded to 30 s like Whisper's feature extractor.
// Greedy decoding after the prompt's token ids; writes the generated ids (no prompt, no end token) and
// returns how many, or -1 with the error in err.
CW_API int cw_transcribe(CwEngine* engine, const float* pcm, int samples, const int* prompt, int prompt_len,
                         int* ids, int max_ids, char* err, int err_len);
// JSON timings of the last load or transcribe call.
CW_API const char* cw_stats(CwEngine* engine);
CW_API void cw_free(CwEngine* engine);

#ifdef __cplusplus
}
#endif
