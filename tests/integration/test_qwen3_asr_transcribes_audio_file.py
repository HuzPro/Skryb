"""Real Qwen3-ASR over a real audio fixture, in a clean subprocess.

The batch adapter is exercised by ``scripts/bench_qwen3_asr.py`` running
in a *fresh* interpreter - the same isolation the streaming integration
tests use, because loading a real speech model alongside pytest's Qt
plugin segfaults on Windows.

Opt-in (``VOICETOTEXT_RUN_REAL_MODEL_TESTS``), GPU-only, skipped when
the Qwen3-ASR snapshot isn't downloaded.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from voicetotext.transcription.hf_model_manager import (
    repo_has_loadable_snapshot,
)

from tests.helpers.gpu import require_cuda
from tests.helpers.model_cache import models_dir


pytestmark = [pytest.mark.integration, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _REPO_ROOT / "scripts" / "bench_qwen3_asr.py"
_REPO_ID = "Qwen/Qwen3-ASR-1.7B-hf"
_OPT_IN = "VOICETOTEXT_RUN_REAL_MODEL_TESTS"


def _skip_unless_qwen3_asr_ready() -> None:
    if not os.environ.get(_OPT_IN):
        pytest.skip(f"set {_OPT_IN} to run real-model tests")
    require_cuda()
    directory = models_dir()
    if directory is None:
        pytest.skip("models dir unavailable")
    repo_dir = directory / f"models--{_REPO_ID.replace('/', '--')}"
    if not repo_has_loadable_snapshot(repo_dir):
        pytest.skip(f"qwen3-asr model not installed: {_REPO_ID}")


def _transcribe_fixture(name: str, *, timeout: float) -> dict[str, str]:
    _skip_unless_qwen3_asr_ready()
    completed = subprocess.run(
        [sys.executable, str(_DRIVER), name],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return _parse_summary(completed.stdout)


def _parse_summary(stdout: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in stdout.splitlines():
        for key in ("FINAL_TEXT:", "TRANSCRIBE_SECONDS:"):
            if line.startswith(key):
                fields[key.rstrip(":")] = line[len(key):].strip()
    return fields


class TestQwen3AsrShortAudio:
    def test_should_transcribe_testing_one_two_three(self):
        summary = _transcribe_fixture("short", timeout=600)

        assert "testing" in summary["FINAL_TEXT"].lower()

    def test_should_report_transcription_seconds(self):
        summary = _transcribe_fixture("short", timeout=600)

        assert float(summary["TRANSCRIBE_SECONDS"]) > 0.0
