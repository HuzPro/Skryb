"""Real Parakeet streaming over real audio fixtures, in a clean subprocess.

The streaming engine is exercised by ``scripts/bench_parakeet_streaming.py``
running in a *fresh* interpreter. That isolation is deliberate: loading
the sentencepiece-backed Parakeet model in the same process as pytest's
Qt plugin segfaults on Windows (a native contamination unrelated to the
streaming logic), exactly the reason the performance suite shells out too.

Opt-in (``VOICETOTEXT_RUN_REAL_MODEL_TESTS``), GPU-only, skipped when the
Parakeet model isn't downloaded. Assertions are lenient — streaming
revises its tail, so we pin behaviours any working run satisfies.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager

from tests.helpers.gpu import require_cuda
from tests.helpers.model_cache import models_dir


pytestmark = [pytest.mark.integration, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _REPO_ROOT / "scripts" / "bench_parakeet_streaming.py"
_REPO_ID = "nvidia/parakeet-tdt-0.6b-v3"
_OPT_IN = "VOICETOTEXT_RUN_REAL_MODEL_TESTS"
_WORD_RE = re.compile(r"[a-z0-9]+")


def _skip_unless_parakeet_ready() -> None:
    if not os.environ.get(_OPT_IN):
        pytest.skip(f"set {_OPT_IN} to run real-model streaming tests")
    require_cuda()
    directory = models_dir()
    if directory is None:
        pytest.skip("models dir unavailable")
    snapshots = (
        directory / f"models--{_REPO_ID.replace('/', '--')}" / "snapshots"
    )
    if not snapshots.exists() or not any(
        HuggingFaceModelManager._snapshot_has_complete_model(snapshot)
        for snapshot in snapshots.iterdir()
    ):
        pytest.skip(f"parakeet model not installed: {_REPO_ID}")


def _stream_fixture(name: str, *extra: str, timeout: float) -> dict[str, str]:
    _skip_unless_parakeet_ready()
    completed = subprocess.run(
        [sys.executable, str(_DRIVER), name, *extra],
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
        for key in ("FINAL_TEXT:", "PARTIAL_COUNT:", "DISTINCT_PARTIALS:"):
            if line.startswith(key):
                fields[key.rstrip(":")] = line[len(key):].strip()
    return fields


class TestParakeetStreamingShortAudio:
    def test_should_transcribe_testing_one_two_three(self):
        summary = _stream_fixture("short", timeout=180)

        assert "testing" in summary["FINAL_TEXT"].lower()

    def test_should_emit_partial_updates_while_streaming(self):
        summary = _stream_fixture("short", timeout=180)

        assert int(summary["PARTIAL_COUNT"]) > 0


class TestParakeetStreamingLongAudio:
    def test_should_emit_distinct_partials_and_many_words(self):
        summary = _stream_fixture("long", "2.0", timeout=600)

        assert int(summary["DISTINCT_PARTIALS"]) >= 2
        assert len(_WORD_RE.findall(summary["FINAL_TEXT"])) > 100
