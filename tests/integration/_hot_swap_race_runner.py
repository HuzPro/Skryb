"""Standalone runner for the record-during-hot-swap characterization.

Lives outside the pytest collection path (leading underscore) and is
invoked as a subprocess from
:mod:`tests.integration.test_recording_during_hot_swap`.  Running the
real :class:`TransformersWhisperAdapter` inside the main pytest
process leaves sentencepiece C++ state alive into interpreter
finalization and segfaults on Windows at shutdown - so we isolate
the whole thing in a child process that calls :func:`os._exit` after
printing a machine-readable result, bypassing normal teardown.

Exit codes
----------
``0``   The race scenario ran and the invariant holds (second
        transcription succeeds using the still-loaded tiny engine).
``2``   The tiny Whisper model is not available in the local cache;
        the pytest wrapper turns this into ``pytest.skip``.
``3``   The invariant was violated - the second transcription failed
        with ``"Model not loaded"`` or any other error.  The pytest
        wrapper turns this into a test failure and surfaces stdout.
``4``   The test harness itself raised unexpectedly (timeout, wiring
        error, etc.).  Treated as a test failure.

The runner prints a single-line JSON blob on stdout summarising the
outcome so the pytest wrapper can attach it to the assertion
message.
"""

import json
import os
import sys
import traceback

from voicetotext.core.exceptions import TranscriptionError

from tests.integration.hot_swap_fixture import HotSwapIntegrationApp


_EXIT_PASSED = 0
_EXIT_MODEL_UNAVAILABLE = 2
_EXIT_INVARIANT_VIOLATED = 3
_EXIT_HARNESS_ERROR = 4


def _run() -> int:
    app = HotSwapIntegrationApp()
    try:
        try:
            app.load_tiny_model_for_real()
        except (OSError, ValueError, TranscriptionError) as exc:
            print(json.dumps({"status": "model_unavailable", "reason": str(exc)}))
            return _EXIT_MODEL_UNAVAILABLE

        app.microphone_will_capture_silence_of(seconds=0.5)

        # Baseline - the real tiny model transcribes silent audio.
        app.user_presses_hotkey()
        app.user_releases_hotkey()
        app.wait_for_next_transcription_outcome(timeout=60)
        if app.transcription_count != 1:
            print(
                json.dumps(
                    {
                        "status": "baseline_failed",
                        "count": app.transcription_count,
                        "errors": app.transcription_errors,
                    }
                )
            )
            return _EXIT_INVARIANT_VIOLATED

        # Reproduce the race: a blocking loader pins the swap thread
        # mid-load so the test can land a hotkey press in the gap.
        app.next_swap_will_use_a_blocking_loader()
        app.user_triggers_engine_swap_to("large-v3")
        app.wait_until_swap_loader_is_blocked(timeout=5)

        app.user_presses_hotkey()               # state -> RECORDING
        app.release_swap_loader()               # loader returns
        app.wait_for_swap_to_finish(timeout=5)  # set_engine raises

        app.user_releases_hotkey()              # state -> TRANSCRIBING
        app.wait_for_next_transcription_outcome(timeout=60)

        if app.transcription_count != 2 or any(
            "Model not loaded" in err for err in app.transcription_errors
        ):
            print(
                json.dumps(
                    {
                        "status": "race_invariant_violated",
                        "count": app.transcription_count,
                        "errors": app.transcription_errors,
                    }
                )
            )
            return _EXIT_INVARIANT_VIOLATED

        print(
            json.dumps(
                {
                    "status": "passed",
                    "count": app.transcription_count,
                    "typed_text": app.typed_text,
                }
            )
        )
        return _EXIT_PASSED
    finally:
        app.shutdown()


def main() -> None:
    try:
        code = _run()
    except Exception:
        traceback.print_exc()
        print(
            json.dumps(
                {"status": "harness_error", "traceback": traceback.format_exc()}
            )
        )
        code = _EXIT_HARNESS_ERROR
    # Bypass normal Python shutdown - sentencepiece's teardown
    # segfaults on Windows after CUDA inference has run.  os._exit
    # lets the parent pytest process see our chosen exit code
    # without the crash.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
