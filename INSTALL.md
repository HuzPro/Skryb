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

On a computer with no GPU, a fresh install defaults to `phonon-2.5-english`: Fermion
Research's 164 MB compression of Parakeet v3 (English only, batch only). It
runs on Fermion's CPU engine from the pinned `fermion-research` package, which
uses native AVX2 / AVX-512 kernels on the packed weights. On a 6-core desktop
forced to the AVX2 kernels, a 3 s dictation took about 0.2 s and long audio ran
about 30x faster than real time. The first load unpacks the 164 MB archive once
into the snapshot's `.phonon2` folder.

### Intel NPU

On a computer with an Intel NPU and no GPU, `pip install -e .[npu]` adds
OpenVINO (the CPU requirements file already includes it). `parakeet-ultra`
then runs its encoder on the NPU, and the rest of the model on the CPU. On a
Core Series 3 (Wildcat Lake) NPU, the one it was tuned and measured on,
`phonon-2.5-english` runs there too by default instead of on Fermion's CPU
engine; on other NPUs pick "Parakeet on the NPU" in Settings to try it. The
first load exports the encoder to OpenVINO IR in the snapshot's `.openvino`
folder, rewrites it with int8 weights and compiles it at five lengths, which
takes a few minutes once; for Phonon, its CPU engine takes dictation
meanwhile and the NPU takes over when every length is up. Later loads reuse
the compiled copies and take dictation in under a second. The NPU needs
Intel's NPU driver 32.0.100.5540 or newer. Without a working NPU the same
engine runs the encoder on the CPU, and Settings says so and why.

The NPU also runs `qwen3-asr-1.7b` (dictation and meetings) and the
meeting-only models, all without torch. Their OpenVINO graphs are
built from `model.safetensors` in the snapshot's `.openvino` folder the first
time, in seconds, and compiled once (under a minute each). Qwen3-ASR's decoder
runs with int8 weights in the NPU plugin's LLM pipeline, and in meetings it
checks Phonon's transcript of each line as a draft, up to 16 tokens a pass,
which keeps the text exactly what Qwen3-ASR alone writes and cuts a 10 s line
from about 2.5 s to 0.6 s. Whisper Hindi2Hinglish Apex (for Urdu meetings)
runs its encoder on the NPU and its decoder on the CPU. The diarizer's model step runs on the NPU, so meetings on
a laptop without a GPU tell speakers apart.
`python scripts\check_npu.py` proves the work reaches the NPU with Windows'
own NPU counters, and races it against the Intel GPU and the CPU.
OpenVINO's model converter reports usage to Intel unless you opt out; Skryb
keeps its telemetry package from loading.

## Streaming dictation (experimental)

Streaming mode types text live as you speak, instead of recording first and transcribing on stop.

- **GPU-only.** A CUDA or ROCm GPU is required; the streaming models are rejected at load on CPU. On a box with no detected GPU the Settings UI disables the Streaming toggle. It is still experimental on AMD ROCm.
- **Parakeet is a core dependency.** Every supported requirements path and a plain `pip install -e .` install `nano-parakeet` (which pulls `sentencepiece`).
- **Models:** `parakeet-ultra` (windowed) and `nemotron-speech-streaming-en-0.6b` (cache-aware, lowest latency). Choose **Streaming** on the Transcription page, then download a streaming model.

## Notes per platform

### Windows AMD (RDNA3 / RDNA4)

- **Python 3.12 only.** AMD does not publish cp310 / cp311 / cp313 Windows ROCm wheels yet. If your venv uses a different interpreter, pip will refuse - recreate it with `py -3.12 -m venv .venv`.
- **AMD Adrenalin driver 26.2.2 or newer** is required for the 7.2.1 wheel.
- The AMD requirements use ROCm 7.2.1. Keep PyTorch and torchaudio on matching ROCm builds when upgrading.
- One wheel covers all RDNA archs - no per-gfx selection needed.
- **`torch.distributed` is unavailable** on the Windows ROCm wheel. Packages that require it at module load will crash on Windows AMD. Don't add `torchao`, `bitsandbytes`, `flash-attn`, `deepspeed`, or `triton` to the project - none of them have Windows ROCm support.

### NVIDIA

- The CUDA wheel index defaults to `cu130`, which covers the RTX 20 series through the RTX 50 series and needs NVIDIA driver 580 or newer. On a GTX 10-series card or older, change it to `cu126` in `requirements-nvidia.txt` (those builds don't run on an RTX 50). PyTorch wheels bundle the CUDA runtime, so only the host driver version matters - no CUDA Toolkit install required.
- The native engines (Parakeet, Qwen3-ASR and the meeting diarizer) have an experimental NVIDIA build for the RTX 30, 40 and 50 series, picked automatically. It checks its own kernels when it loads and falls back to the CPU if they disagree.

### No supported GPU or NPU

- Meetings still tell speakers apart, through an experimental CPU build of the same diarizer (NVIDIA's NeMo-Speech.cpp). It downloads a 107 MB model the first time and keeps up with live audio on a recent desktop CPU. If the CPU falls behind, the speaker labels finish after the meeting ends instead.

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

### "Could not find a version that satisfies the requirement torch==2.9.1+rocm7.2.1"

You're on the wrong Python version for the AMD Windows wheels. They're cp312 only. Recreate the venv with `py -3.12 -m venv .venv`.
