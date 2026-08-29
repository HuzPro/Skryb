"""Transcribe the short audio fixture with the real CrisperWhisper adapter.

This clean-process driver doubles as the subprocess target for the opt-in
integration test. It never downloads weights: the adapter requires the turbo
snapshot at the catalog's pinned commit to already exist in VoiceToText's
model cache. CrisperWhisper's licence covers use and generated outputs, so
the run refuses to start until the licence is accepted in the app.

    python scripts/bench_crisperwhisper.py intended
    python scripts/bench_crisperwhisper.py verbatim

The final ``MODE:`` and ``FINAL_TEXT:`` lines are machine-readable.
"""

import os
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Must precede any ``import torch`` so the ROCm SDPA path is enabled.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


_MODEL_NAME = "crisperwhisper-2.0-turbo"


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    from tests.fixtures import SHORT_MP3
    from tests.helpers.audio_file_loader import load_audio_file
    from tests.helpers.license_gate import user_license_policy
    from voicetotext.core.paths import AppPaths
    from voicetotext.transcription.crisperwhisper_adapter import (
        CrisperWhisperAdapter,
        resolve_crisperwhisper_mode,
    )
    from voicetotext.transcription.model_info import get_model_info

    mode = resolve_crisperwhisper_mode(
        sys.argv[1] if len(sys.argv) > 1 else "intended"
    )
    model_info = get_model_info(_MODEL_NAME)
    user_license_policy().assert_accepted(model_info)
    audio = load_audio_file(SHORT_MP3)
    engine = CrisperWhisperAdapter(
        repo_id=model_info.repo_id,
        revision=model_info.revision,
        device="cuda",
        model_dir=AppPaths().models_dir,
        language="en",
        mode=mode,
    )

    load_started = time.monotonic()
    engine.load_model()
    log(f"model load: {time.monotonic() - load_started:.2f}s")
    result = engine.transcribe(audio)
    engine.dispose()

    log(f"MODE: {mode}")
    log(f"FINAL_TEXT: {result.text}")


if __name__ == "__main__":
    main()
