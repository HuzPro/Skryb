# Native engines (RX 9060 XT, gfx1200; NVIDIA RTX 30-50, experimental)

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

## NVIDIA build

The same sources build for NVIDIA through `engine/gpu.h` (HIP's runtime names mapped onto CUDA's) and `engine/tile.h`
(the 16x16x16 bf16 tile product: gfx12 WMMA on AMD, two `mma.sync` m16n8k16 on NVIDIA). One DLL per engine
(`qasr_cuda.dll`, `npk_cuda.dll`, `cw_cuda.dll`, `nd_cuda.dll`) carries code for sm_86 (RTX 30), sm_89 (RTX 40) and
sm_120 (RTX 50) plus PTX, with the CUDA runtime linked in: users need only the NVIDIA driver. No GPU needed to build:

```bash
python native/cuda/fetch_toolkit.py   # nvcc, the runtime and CCCL from NVIDIA's archives, into native/cuda/build
python native/cuda/build.py           # all four, next to the AMD ones in native/*/build
```

Every engine runs `gpu_self_check` at load: the GEMM paths and attention on small random inputs against a CPU
reference. A GPU the kernels misbehave on fails to load (and Skryb retries on the CPU) instead of writing nonsense.

`-DMMA_NV_LAYOUT` builds the NVIDIA tile layout for AMD, with the matrix product done by shuffles, so everything but
the two NVIDIA instructions can be checked on the RX 9060 XT. On the bench corpus it matches the AMD build to bf16
rounding.
