"""Transcribe an audio fixture with the real ARK-ASR-3B adapter.

Runs in a clean process (no Qt), so it doubles as the subprocess runner
for the opt-in integration test and as a human smoke-check tool.

    python scripts/bench_ark_asr.py short
    python scripts/bench_ark_asr.py long

The arg is the fixture (``short`` = Testing_1_2_3, ``long`` =
LongStory). Machine-readable summary lines (``FINAL_TEXT:`` /
``TRANSCRIBE_SECONDS:``) are printed last for the test to parse.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must precede any ``import torch`` so the ROCm attention path is enabled.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import time

_MODEL_NAME = "ark-asr-3b"


def log(message: str) -> None:
    print(message, flush=True)


def _fixture_path(name: str) -> Path:
    from tests.fixtures import LONG_MP3, SHORT_MP3

    return {"short": SHORT_MP3, "long": LONG_MP3}[name]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "short"

    from tests.helpers.audio_file_loader import load_audio_file
    from voicetotext.core.paths import AppPaths
    from voicetotext.transcription.ark_asr_adapter import ArkAsrAdapter
    from voicetotext.transcription.model_info import get_model_info

    audio = load_audio_file(_fixture_path(name))
    log(f"clip={name} duration={audio.duration_seconds:.2f}s")

    model_info = get_model_info(_MODEL_NAME)
    adapter = ArkAsrAdapter(
        repo_id=model_info.repo_id,
        revision=model_info.revision,
        device="cuda",
        model_dir=AppPaths().models_dir,
    )
    load_started = time.monotonic()
    adapter.load_model()
    log(f"model load: {time.monotonic() - load_started:.2f}s")

    warmup_started = time.monotonic()
    adapter.warmup()
    log(f"warmup: {time.monotonic() - warmup_started:.2f}s")

    started = time.monotonic()
    result = adapter.transcribe(audio)
    elapsed = time.monotonic() - started
    adapter.dispose()

    log(f"FINAL_TEXT: {result.text}")
    log(f"TRANSCRIBE_SECONDS: {elapsed:.2f}")


if __name__ == "__main__":
    main()
