"""Opt-in real CrisperWhisper inference in a clean subprocess.

The test is GPU-only and requires the turbo model to have already been
downloaded through Skryb at the catalog's pinned commit. It does
not contact Hugging Face. The driver it launches refuses to run until
the model's licence has been accepted in the app.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.fixtures import require_short_fixture
from tests.helpers.gpu import require_cuda
from tests.helpers.model_cache import models_dir
from skryb.transcription.hf_model_manager import (
    repo_has_loadable_snapshot,
)
from skryb.transcription.model_info import get_model_info


pytestmark = [pytest.mark.integration, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _REPO_ROOT / "scripts" / "bench_crisperwhisper.py"
_CRISPERWHISPER = get_model_info("crisperwhisper-2.0-turbo")
_OPT_IN = "SKRYB_RUN_REAL_MODEL_TESTS"


def _skip_unless_crisperwhisper_ready() -> None:
    if not os.environ.get(_OPT_IN):
        pytest.skip(f"set {_OPT_IN} to run real-model tests")
    require_cuda()
    require_short_fixture()
    directory = models_dir()
    if directory is None:
        pytest.skip("models dir unavailable")
    # Any-snapshot would let this validate stale pre-pin weights; the
    # loader opens the pinned commit and nothing else.
    repo_dir = (
        directory
        / f"models--{_CRISPERWHISPER.repo_id.replace('/', '--')}"
    )
    if not repo_has_loadable_snapshot(repo_dir, _CRISPERWHISPER.revision):
        pytest.skip(
            "CrisperWhisper pinned snapshot not installed: "
            f"{_CRISPERWHISPER.repo_id}"
        )


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
