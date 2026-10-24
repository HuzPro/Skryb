# Native Qwen3-ASR engine (RX 9060 XT, gfx1200)

Hand-written HIP kernels behind `qasr.dll`, loaded by
`skryb/transcription/native_qwen3_asr_adapter.py` for the `qwen3-asr-1.7b` model. No torch.

Build (needs AMD's `rocm-sdk-core` and `rocm-sdk-devel` 7.2.1 wheels in the Python 3.12 on
`PATH`, plus Visual Studio 2022's C++ build tools; no GPU needed). Visual Studio 2026's C++ library
(MSVC 14.51) doesn't compile with ROCm 7.2.1's HIP headers, so the Release workflow builds on `windows-2022`:

```bash
python -m pip install --no-deps --find-links https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/ rocm-sdk-core==7.2.1 rocm-sdk-devel==7.2.1
bash native/engine/build.sh
```

The Release workflow does the same and ships `qasr.dll` inside the app wheel as
`skryb/transcription/qasr.dll`. The app loads that copy, then `native/engine/build/qasr.dll` in a
source checkout; `QASR_DLL` overrides both. The kernels are compiled for gfx1200 only, so the
`qwen3-asr-1.7b` model is offered only on a gfx1200 GPU with the DLL present.

Run HIP binaries from a short path: the HIP runtime crashes (0xC0000409) when its module path is long.

Benchmark against the PyTorch reference (clips stay outside the repo, in `D:/VoiceToText_bench`):

```bash
python native/bench/corpus.py
<venv with ROCm torch + transformers>/python native/bench/reference.py --model <Qwen3-ASR-1.7B-hf dir>
native/engine/build/qasr-bench.exe <snapshot>/.qasr/qwen3-asr.bin D:/VoiceToText_bench/corpus.jsonl D:/VoiceToText_bench/native.jsonl
python native/bench/compare.py
venv/Scripts/python.exe native/bench/app_engine_check.py
```

`QASR_DRAFT=0` turns off speculative decoding; `QASR_PROFILE=1` adds per-op prefill times to the stats.
