"""Stream an audio fixture through Parakeet and print partials over time.

Runs in a clean process (no Qt), so it doubles as the subprocess runner
for the streaming integration test and as a human "watch it stream" tool.

    python scripts/bench_parakeet_streaming.py short
    python scripts/bench_parakeet_streaming.py long 2.0

The first arg is the fixture (``short`` = Testing_1_2_3, ``long`` =
LongStory); the optional second arg is the decode cadence in seconds.
Machine-readable summary lines (``FINAL_TEXT:`` / ``PARTIAL_COUNT:`` /
``DISTINCT_PARTIALS:``) are printed last for the test to parse.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must precede any ``import torch`` so the ROCm attention path is enabled.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import time

_REPO_ID = "nvidia/parakeet-tdt-0.6b-v3"
_CHUNK_SECONDS = 0.25


def log(message: str) -> None:
    print(message, flush=True)


def _fixture_path(name: str) -> Path:
    from tests.fixtures import LONG_MP3, SHORT_MP3

    return {"short": SHORT_MP3, "long": LONG_MP3}[name]


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "short"
    cadence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6

    from voicetotext.core.paths import AppPaths
    from voicetotext.transcription.streaming.audio_chunker import iter_chunks
    from voicetotext.transcription.streaming.parakeet_streaming_engine import (
        StreamingNanoParakeetEngine,
    )
    from tests.helpers.audio_file_loader import load_audio_file

    audio = load_audio_file(_fixture_path(name))
    log(f"clip={name} duration={audio.duration_seconds:.2f}s cadence={cadence}s")

    engine = StreamingNanoParakeetEngine(
        _REPO_ID, AppPaths().models_dir, device="cuda", cadence_seconds=cadence,
    )
    load_started = time.monotonic()
    engine.load_model()
    log(f"model load: {time.monotonic() - load_started:.2f}s")

    session = engine.start_session()
    started = time.monotonic()
    distinct: set[str] = set()
    count = 0
    last_printed = None
    for chunk in iter_chunks(audio, _CHUNK_SECONDS):
        update = session.feed(chunk)
        count += 1
        if update.text.strip():
            distinct.add(update.text)
        printable = (update.committed_text, update.tentative_text)
        if printable != last_printed:
            log(
                f"  [{time.monotonic() - started:6.2f}s] "
                f"committed={update.committed_text!r} "
                f"tentative={update.tentative_text!r}"
            )
            last_printed = printable
    result = session.finalize()
    engine.dispose()

    log(f"FINAL_TEXT: {result.text}")
    log(f"PARTIAL_COUNT: {count}")
    log(f"DISTINCT_PARTIALS: {len(distinct)}")


if __name__ == "__main__":
    main()
