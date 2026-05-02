"""Characterization test for the record-during-hot-swap race.

Reproduces the production bug where starting a recording while an
engine hot-swap is loading a new model would leave
``RecordAndTranscribeUseCase`` referencing a disposed engine, and
every subsequent recording would fail with
``TranscriptionError("Model not loaded")``.

The actual scenario is driven by
:mod:`tests.integration._hot_swap_race_runner` in a subprocess.  That
child process uses the **real** :class:`TransformersWhisperAdapter`
with the tiny Whisper model so the adapter's post-dispose behaviour
(raising ``"Model not loaded"``) is observable — a fake engine would
silently mask the very failure the test exists to catch.

Running the real transformers stack inside pytest leaks sentencepiece
C++ state into interpreter finalization and segfaults on Windows at
shutdown.  Isolating the scenario in a child process that calls
``os._exit`` after printing its result keeps the parent pytest
process clean.

The whole test is gated behind ``VOICETOTEXT_RUN_REAL_MODEL_TESTS`` —
without that env var the test skips before ever spawning the
subprocess. On Windows, ``torch`` opens shared-memory regions backed
by the OS pagefile during import, and the auto-expanding pagefile
can fill the disk if the subprocess is spawned while resources are
already low. Opt in only when you have headroom.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = (
    _REPO_ROOT / "tests" / "integration" / "_hot_swap_race_runner.py"
)

_EXIT_PASSED = 0
_EXIT_MODEL_UNAVAILABLE = 2
_EXIT_INVARIANT_VIOLATED = 3
_EXIT_HARNESS_ERROR = 4

_REAL_MODEL_OPT_IN = "VOICETOTEXT_RUN_REAL_MODEL_TESTS"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get(_REAL_MODEL_OPT_IN),
    reason=(
        "Real-model integration test gated behind "
        f"{_REAL_MODEL_OPT_IN}=1 — opting in spawns a subprocess that "
        "imports torch and can balloon the Windows pagefile to fill "
        "the disk."
    ),
)
class TestRecordingDuringHotSwap:
    def test_old_engine_keeps_transcribing_when_recording_races_with_swap(
        self,
    ) -> None:
        result = subprocess.run(
            [sys.executable, str(_RUNNER)],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        report = self._parse_runner_output(result.stdout)

        if result.returncode == _EXIT_MODEL_UNAVAILABLE:
            pytest.skip(
                f"tiny Whisper model not available in local cache: "
                f"{report.get('reason', 'unknown')}"
            )

        assert result.returncode == _EXIT_PASSED, (
            f"hot-swap race runner exited with code {result.returncode}.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}\n"
            f"parsed: {report}"
        )

    def _parse_runner_output(self, stdout: str) -> dict:
        for line in reversed(stdout.strip().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {}
