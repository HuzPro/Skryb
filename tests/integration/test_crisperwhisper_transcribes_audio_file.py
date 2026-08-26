"""Opt-in real CrisperWhisper inference in a clean subprocess.

The test is GPU-only and requires the turbo model to have already been
downloaded through VoiceToText. It does not contact Hugging Face.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures import require_short_fixture
from tests.helpers.gpu import require_cuda
from tests.helpers.model_cache import models_dir
from voicetotext.transcription.hf_model_manager import (
    HuggingFaceModelManager,
)


pytestmark = [pytest.mark.integration, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _REPO_ROOT / "scripts" / "bench_crisperwhisper.py"
_REPO_ID = "nyralabs/CrisperWhisper2.0_turbo"
_OPT_IN = "VOICETOTEXT_RUN_REAL_MODEL_TESTS"


def _skip_unless_crisperwhisper_ready() -> None:
    if not os.environ.get(_OPT_IN):
        pytest.skip(f"set {_OPT_IN} to run real-model tests")
    require_cuda()
    require_short_fixture()
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
        pytest.skip(f"CrisperWhisper model not installed: {_REPO_ID}")


def _transcribe_in_subprocess(mode: str) -> str:
    _skip_unless_crisperwhisper_ready()
    completed = subprocess.run(
        [sys.executable, str(_DRIVER), mode],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for line in completed.stdout.splitlines():
        if line.startswith("FINAL_TEXT:"):
            return line.removeprefix("FINAL_TEXT:").strip()
    raise AssertionError(f"FINAL_TEXT missing from output:\n{completed.stdout}")


@pytest.mark.parametrize("mode", ["intended", "verbatim"])
def test_should_transcribe_short_fixture_in_each_mode(mode: str) -> None:
    text = _transcribe_in_subprocess(mode)

    assert "testing" in text.lower()
