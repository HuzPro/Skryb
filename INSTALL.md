# Install VoiceToText (Windows / Linux)

## Prerequisites

- **Python 3.12** (required for the AMD Windows path; 3.11+ works for everything else)
- Git

## Pick a requirements file

| File | Use when |
|------|----------|
| `requirements-amd.txt`       | Windows AMD GPU. Installs PyTorch ROCm 7.2.1 + AMD's TheRock runtime from `repo.radeon.com`. |
| `requirements-amd-linux.txt` | Linux AMD GPU. Same ROCm version, AMD's manylinux index. |
| `requirements-nvidia.txt`    | NVIDIA GPU (Windows or Linux). Installs PyTorch CUDA build. |
| `requirements-cpu.txt`       | No GPU (or smaller download). Installs CPU-only PyTorch. |
| `requirements-all.txt`       | You already have PyTorch installed and just want the rest of the deps. |
| `requirements.txt`           | Same as `requirements-all.txt` minus `faster-whisper` and `nano-parakeet`. |

The GPU files use `--find-links` (AMD's index is a flat HTML directory) or `--extra-index-url` (PyTorch's CUDA index is a PEP 503 index) to pull the right wheel without disturbing PyPI lookups for everything else.

## Steps

```powershell
git clone <repo-url> VoiceToText
cd VoiceToText

py -3.12 -m venv .venv             # Python 3.12 explicitly for AMD Windows
.\.venv\Scripts\Activate.ps1       # Windows
# source .venv/bin/activate        # Linux / macOS

pip install -r requirements-amd.txt   # or -amd-linux / -nvidia / -cpu / -all
python -m voicetotext
```

First run opens a setup window where you pick a model and download it.

## Streaming dictation (experimental)

Streaming mode types text live as you speak, instead of recording first and transcribing on stop.

- **GPU-only.** A CUDA or ROCm GPU is required; the streaming models are rejected at load on CPU. On a box with no detected GPU the Settings UI disables the Streaming toggle. It is still experimental on AMD ROCm.
- **Needs the Parakeet extra.** Install via `requirements-all.txt` / `-amd.txt` / `-nvidia.txt` — these include `nano-parakeet` (which pulls `sentencepiece`). The bare `requirements.txt` omits it and streaming will not load.
- **Models:** `parakeet-tdt-0.6b-v3` (windowed) and `nemotron-speech-streaming-en-0.6b` (cache-aware, lowest latency). Choose **Streaming** on the Transcription page, then download a streaming model.

## Notes per platform

### Windows AMD (RDNA3 / RDNA4)

- **Python 3.12 only.** AMD does not publish cp310 / cp311 / cp313 Windows ROCm wheels yet. If your venv uses a different interpreter, pip will refuse — recreate it with `py -3.12 -m venv .venv`.
- **AMD Adrenalin driver 26.2.2 or newer** is required for the 7.2.1 wheel.
- ROCm 7.2.1 is the latest stable Windows release (March 2026). To upgrade later, update the version strings inside `requirements-amd.txt` to the new `rocm-rel-X.Y.Z` directory at https://repo.radeon.com/rocm/windows/.
- One wheel covers all RDNA archs — no per-gfx selection needed.
- **`torch.distributed` is unavailable** on the Windows ROCm wheel — anything that imports `torch._C._distributed_c10d` at module load (notably `transformers>=5.9`, `torchao`, `deepspeed`, `bitsandbytes`) will crash on Windows AMD. The requirements files cap `transformers<5.9` for this reason. Don't add `torchao`, `bitsandbytes`, `flash-attn`, `deepspeed`, or `triton` to the project — none of them have Windows ROCm support.

### Linux AMD

- Python 3.10 / 3.11 / 3.12 / 3.13 all supported.
- Linux ROCm wheels carry a `.lw.git<sha>` local-version suffix on top of `+rocm7.2.1`, so the Linux file pins `torch>=2.9.1,<2.10` instead of an exact `==` match.

### NVIDIA

- The CUDA wheel index defaults to `cu126`. Edit the first line of `requirements-nvidia.txt` to `cu124` / `cu121` / `cu118` if you want a different CUDA runtime. PyTorch wheels bundle the CUDA runtime, so only the host driver version matters — no CUDA Toolkit install required.

### "I already have a custom torch installed"

Use `requirements-all.txt` — it leaves torch alone and only installs the rest of the deps. This is what to do if you have a nightly ROCm / CUDA build, or an internal corporate index, that the GPU-specific files would clobber.

## Troubleshooting

If transcription returns an empty clipboard or the tray icon stays on **Loading**, run:

```powershell
python -m voicetotext --doctor
```

Output lists Python / PyTorch versions, which engine backends are installed, CUDA / ROCm device availability, and the HuggingFace cache path. Share that output when reporting an issue — most missing-dependency or device-detection problems jump out from one line of the report.

### "Cannot install torch>=X because these package versions have conflicting dependencies"

You have a custom torch already (different ROCm / CUDA / nightly version) and you're using the wrong file. Switch to `requirements-all.txt` so pip leaves torch alone.

### "Could not find a version that satisfies the requirement torch==2.9.1+rocm7.2.1"

You're on the wrong Python version for the AMD Windows wheels. They're cp312 only. Recreate the venv with `py -3.12 -m venv .venv`.

### "ModuleNotFoundError: No module named 'torch._C._distributed_c10d'"

You have `transformers>=5.9` installed alongside AMD's Windows ROCm torch. Transformers 5.9+ added an unconditional import of `torch.distributed.tensor`, which doesn't exist in AMD's Windows build. Downgrade transformers:

```powershell
pip install "transformers>=4.52.1,<5.9"
```

Or simply re-run `pip install -r requirements-amd.txt` after pulling the latest changes — the requirements files now pin `transformers<5.9`.

### "Using a device_map ... requires accelerate"

`accelerate` is a soft-dependency of transformers needed whenever a model is loaded with `device_map=...` (which the Granite Speech adapter does for direct-to-GPU loading). Install it:

```powershell
pip install "accelerate>=1.1.0"
```

The requirements files now declare it explicitly, so a fresh `pip install -r requirements-amd.txt` will pull it.

### "GraniteSpeechFeatureExtractor requires the torchaudio library"

Granite Speech's feature extractor builds its mel-spectrogram with `torchaudio`, so it's a hard dependency of that engine. Earlier `requirements-nvidia.txt` / `requirements-cpu.txt` shipped `torch` alone, so NVIDIA/CPU users hit this at transcription time. Install the matching build:

```powershell
# NVIDIA (match the CUDA index from requirements-nvidia.txt)
pip install torchaudio --extra-index-url https://download.pytorch.org/whl/cu126
# CPU
pip install torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

Or simply re-run `pip install -r requirements-nvidia.txt` (or `-cpu`) after pulling the latest changes — all one-shot files now pull `torchaudio` alongside `torch`.
