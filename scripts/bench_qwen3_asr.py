"""Transcribe an audio fixture through the real Qwen3-ASR batch adapter.

Runs in a clean process (no Qt), so it doubles as the subprocess runner
for the Qwen3-ASR integration test and as a human "hear it transcribe"
tool. The sibling ``bench_*_streaming.py`` scripts drive the streaming
engines; this one exercises the batch :class:`Qwen3AsrAdapter` that
``qwen3_asr`` models route to.

    python scripts/bench_qwen3_asr.py short
    python scripts/bench_qwen3_asr.py long

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

_REPO_ID = "Qwen/Qwen3-ASR-1.7B-hf"


def log(message: str) -> None:
    print(message, flush=True)


def _fixture_path(name: str) -> Path:
    from tests.fixtures import LONG_MP3, SHORT_MP3

    return {"short": SHORT_MP3, "long": LONG_MP3}[name]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "short"

    from tests.helpers.audio_file_loader import load_audio_file
    from voicetotext.core.paths import AppPaths
    from voicetotext.transcription.qwen3_asr_adapter import Qwen3AsrAdapter

    audio = load_audio_file(_fixture_path(name))
    log(f"clip={name} duration={audio.duration_seconds:.2f}s")

    adapter = Qwen3AsrAdapter(
        repo_id=_REPO_ID,
        device="cuda",
        model_dir=AppPaths().models_dir,
        language="en",
    )
    load_started = time.monotonic()
    adapter.load_model()
    log(f"model load: {time.monotonic() - load_started:.2f}s")

    adapter.warmup()
    result = adapter.transcribe(audio)
    adapter.dispose()

    log(f"FINAL_TEXT: {result.text}")
    log(f"TRANSCRIBE_SECONDS: {result.transcription_seconds:.2f}")


if __name__ == "__main__":
    main()
