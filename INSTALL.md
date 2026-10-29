# Install Skryb on Windows

## Prerequisites

Most people should use the installer from the
[latest release](https://github.com/HuzPro/Skryb/releases/latest). This
page is for running from a source checkout.

- **Python 3.12** (required for the AMD Windows path; 3.11+ works for everything else)
- Git
- Windows 10 or 11, 64-bit. The tray, hotkeys, and text output require Windows.
- [Rust](https://rustup.rs), Node 22 and pnpm 11, to build the settings window once.

## Pick a requirements file

| File | Use when |
|------|----------|
| `requirements-amd.txt`       | Windows AMD GPU. Installs PyTorch ROCm 7.2.1 + AMD's TheRock runtime from `repo.radeon.com`. |
| `requirements-nvidia.txt`    | Windows NVIDIA GPU. Installs PyTorch CUDA build. |
| `requirements-cpu.txt`       | No GPU (or smaller download). Installs CPU-only PyTorch. |
| `requirements-all.txt`       | Core dependencies for an environment with a matching PyTorch runtime. |
| `requirements.txt`           | Core dependencies, including the `nano-parakeet` engine, without PyTorch. |

The GPU files use `--find-links` (AMD's index is a flat HTML directory) or `--extra-index-url` (PyTorch's CUDA index is a PEP 503 index) to pull the right wheel without disturbing PyPI lookups for everything else.

## Steps

```powershell
git clone https://github.com/HuzPro/Skryb.git
cd Skryb

py -3.12 -m venv .venv             # Python 3.12 explicitly for AMD Windows
.\.venv\Scripts\Activate.ps1       # Windows

pip install -r requirements-amd.txt   # or requirements-nvidia.txt / requirements-cpu.txt
pip install --no-deps -e .
.\scripts\build_ui.ps1             # builds the settings window, about a minute
python -m skryb
```

Run `scripts\build_ui.ps1` again after pulling changes under `ui\`.

First run opens a setup window where you pick a model and download it.

The editable install above adds both the
console command and a console-free `skryb-gui.exe`. Run
`skryb --create-shortcut` to add **Skryb (Dev)** to the Start
Menu. `skryb --enable-autostart` and `--disable-autostart` manage the
development install's per-user startup entry. Installer uninstall cleanup for
that development entry is intentionally out of scope.

Parakeet Ultra (`parakeet-ultra`, Moondream's retrained Parakeet v3: same 25
languages, fewer errors, half the download) is the batch model and the
fresh-install default on a GPU. Nemotron (`nemotron-speech-streaming-en-0.6b`)
is available in streaming mode. A CUDA or
ROCm GPU is expected for those. Parakeet technically runs on CPU, but measured
throughput is about 15x slower than real time.

On a computer with no GPU, a fresh install defaults to `phonon-2`: Fermion
Research's 164 MB compression of Parakeet v3 (English only, batch only). It
runs on Fermion's CPU engine from the pinned `fermion-research` package, which
uses native AVX2 / AVX-512 kernels on the packed weights. On a 6-core desktop
forced to the AVX2 kernels, a 3 s dictation took about 0.2 s and long audio ran
about 30x faster than real time. The first load unpacks the 164 MB archive once
into the snapshot's `.phonon2` folder.

Underdog Bark 3.0 (Conway Research, CC-BY-4.0, English only) is Parakeet v3
retrained and packed into one `model.bark`: `underdog-bark-3.0` (163 MB),
`underdog-bark-3.0-plus` (180 MB, fewer errors), `underdog-bark-3.0-clean`
(180 MB, also drops "um", false starts and repeats) and `underdog-bark-3.0-pro`
(1.25 GB, the most accurate, plain safetensors). They run on the Parakeet
engine. The first load unpacks `model.bark` once into the snapshot's `.bark`
folder (about 1.3 GB), using the pinned `constriction` package.

### Intel NPU

On a computer with an Intel NPU and no GPU, `pip install -e .[npu]` adds
OpenVINO (the CPU requirements file already includes it). The Bark models and
`parakeet-ultra` then run their encoder on the NPU, and the rest of the model
on the CPU. The first load exports the encoder to OpenVINO IR in the snapshot's
`.openvino` folder and compiles it, which takes a minute or two; later loads
reuse the compiled copy. The NPU needs Intel's NPU driver 32.0.100.5540 or
newer. Without a working NPU the same engine runs the encoder on the CPU.
`python scripts\check_npu.py` reports which device it got and how fast it is.
OpenVINO's model converter reports usage to Intel unless you opt out; Skryb
keeps its telemetry package from loading.

## Streaming dictation (experimental)

Streaming mode types text live as you speak, instead of recording first and transcribing on stop.

- **GPU-only.** A CUDA or ROCm GPU is required; the streaming models are rejected at load on CPU. On a box with no detected GPU the Settings UI disables the Streaming toggle. It is still experimental on AMD ROCm.
- **Parakeet is a core dependency.** Every supported requirements path and a plain `pip install -e .` install `nano-parakeet` (which pulls `sentencepiece`).
- **Models:** `parakeet-ultra` (windowed) and `nemotron-speech-streaming-en-0.6b` (cache-aware, lowest latency). Choose **Streaming** on the Transcription page, then download a streaming model.

## CTranslate2 (fast CrisperWhisper launch, optional)

CrisperWhisper runs on PyTorch + Transformers out of the box. Importing
Transformers costs 10.5 s of a 15.5 s cold start in a development benchmark. The same
model under CTranslate2 reaches ready in 3.4 s and produces byte-identical
transcripts in both intended and verbatim modes, so this is worth
installing if you dictate with CrisperWhisper.

Installing it changes nothing else: no other engine uses CTranslate2, and
without it CrisperWhisper keeps working on Transformers.

**Uninstall any CTranslate2 you already have first.**

```powershell
pip uninstall -y ctranslate2
```

Two reasons, both real. A CPU/CUDA CTranslate2 in a ROCm PyTorch venv
aborts the process (`OMP: Error #15`, no Python traceback) the moment it
loads a model, because it brings its own Intel OpenMP alongside PyTorch's
LLVM one. And every build carries the same `4.8.2` version, so pip
considers an already-installed wrong wheel to satisfy the requirement and
leaves it in place.

### NVIDIA and CPU

```powershell
pip install "ctranslate2>=4.8.2,<5"      # or: pip install -e .[ct2]
```

### AMD (Windows)

The ROCm build is not on PyPI. It ships as a release asset on
CTranslate2's GitHub release, `rocm-python-wheels-Windows.zip` under
https://github.com/OpenNMT/CTranslate2/releases/tag/v4.8.2, holding one
wheel: `temp-windows/ctranslate2-4.8.2-cp312-cp312-win_amd64.whl`. Python
3.12 only, same as the AMD PyTorch wheels.

```powershell
# after downloading and extracting the zip
pip install .\temp-windows\ctranslate2-4.8.2-cp312-cp312-win_amd64.whl
```

The wheel needs `amdhip64_7.dll` and `hipblas.dll`, which
`requirements-amd.txt` already installs via `rocm-sdk-core` and
`rocm-sdk-libraries-custom`. Nothing else to wire up.

### Check you got the right wheel

```powershell
python -m skryb --doctor
```

The `[ctranslate2]` section must report at least one GPU device on a GPU
machine:

```
[ctranslate2]
  version: 4.8.2
  gpu devices: 1
  gpu compute types: bfloat16, float16, float32, int8, ...
```

`gpu devices: 0` on a machine where the `[torch]` section lists a GPU
means the wrong wheel is installed. Uninstall and repeat.

### Disk cost

The first time a CrisperWhisper model is downloaded, the app builds a
CTranslate2 copy of it next to the weights (about 9 s of CPU work, once).
That copy is a second full set of weights: roughly 1.6 GB for
`crisperwhisper-2.0-turbo` and 3.1 GB for `crisperwhisper-2.0-large`, on
top of the download. Deleting the model in Settings deletes both. If you
already downloaded a CrisperWhisper model before installing CTranslate2,
the copy is built on the next engine load instead, which makes that one
launch about 9 s slower.

## Notes per platform

### Windows AMD (RDNA3 / RDNA4)

- **Python 3.12 only.** AMD does not publish cp310 / cp311 / cp313 Windows ROCm wheels yet. If your venv uses a different interpreter, pip will refuse - recreate it with `py -3.12 -m venv .venv`.
- **AMD Adrenalin driver 26.2.2 or newer** is required for the 7.2.1 wheel.
- The AMD requirements use ROCm 7.2.1. Keep PyTorch and torchaudio on matching ROCm builds when upgrading.
- One wheel covers all RDNA archs - no per-gfx selection needed.
- **`torch.distributed` is unavailable** on the Windows ROCm wheel. Packages that require it at module load will crash on Windows AMD. Don't add `torchao`, `bitsandbytes`, `flash-attn`, `deepspeed`, or `triton` to the project - none of them have Windows ROCm support.

### NVIDIA

- The CUDA wheel index defaults to `cu126`. Edit the first line of `requirements-nvidia.txt` to `cu124` / `cu121` / `cu118` if you want a different CUDA runtime. PyTorch wheels bundle the CUDA runtime, so only the host driver version matters - no CUDA Toolkit install required.

### "I already have a custom torch installed"

Use matching PyTorch and torchaudio builds, then install `requirements-all.txt` with a constraints file pinning those versions. Although that file does not list torch directly, torchaudio depends on it and pip may otherwise replace your runtime.

## Troubleshooting

If transcription returns an empty clipboard or the tray icon stays on **Loading**, run:

```powershell
python -m skryb --doctor
```

Output lists Python / PyTorch versions, which engine backends are installed, CUDA / ROCm device availability, and the HuggingFace cache path. Share that output when reporting an issue - most missing-dependency or device-detection problems jump out from one line of the report.

### "Cannot install torch>=X because these package versions have conflicting dependencies"

Check that PyTorch and torchaudio use matching versions and runtimes. Use the matching platform requirements in a fresh environment, or provide constraints for your custom builds.

### The app dies with no traceback, or `OMP: Error #15`

A CTranslate2 built for the wrong runtime is installed next to a ROCm
PyTorch. Run `pip uninstall -y ctranslate2`, then reinstall following the
CTranslate2 section above. `python -m skryb --doctor` tells you
which build you have.

### "Could not find a version that satisfies the requirement torch==2.9.1+rocm7.2.1"

You're on the wrong Python version for the AMD Windows wheels. They're cp312 only. Recreate the venv with `py -3.12 -m venv .venv`.
