"""Real ARK-ASR-3B over real audio, in a clean subprocess.

Drives the production batch path (``ark-asr-3b`` -> :class:`ArkAsrAdapter`
-> transformers remote code) end to end via ``scripts/bench_ark_asr.py``
in a *fresh* interpreter - the same isolation the Parakeet and nemotron
streaming tests use, because loading a real speech model alongside
pytest's Qt plugin segfaults on Windows.

Opt-in (``VOICETOTEXT_RUN_REAL_MODEL_TESTS``), GPU-only, skipped when
the pinned snapshot isn't downloaded.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.gpu import require_cuda
from tests.helpers.model_cache import models_dir
from voicetotext.transcription.hf_model_manager import repo_has_loadable_snapshot
from voicetotext.transcription.model_info import get_model_info


pytestmark = [pytest.mark.integration, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _REPO_ROOT / "scripts" / "bench_ark_asr.py"
_ARK_ASR = get_model_info("ark-asr-3b")
_OPT_IN = "VOICETOTEXT_RUN_REAL_MODEL_TESTS"
_WORD_RE = re.compile(r"[a-z0-9]+")


def _skip_unless_ark_asr_ready() -> None:
    if not os.environ.get(_OPT_IN):
        pytest.skip(f"set {_OPT_IN} to run real-model tests")
    require_cuda()
    directory = models_dir()
    if directory is None:
        pytest.skip("models dir unavailable")
    repo_dir = directory / f"models--{_ARK_ASR.repo_id.replace('/', '--')}"
    if not repo_has_loadable_snapshot(repo_dir, _ARK_ASR.revision):
        pytest.skip(
            f"ARK-ASR pinned snapshot not installed: {_ARK_ASR.repo_id}"
        )


def _transcribe_fixture(name: str, *, timeout: float) -> dict[str, str]:
    _skip_unless_ark_asr_ready()
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


class TestArkAsrShortAudio:
    def test_should_transcribe_testing_one_two_three(self):
        summary = _transcribe_fixture("short", timeout=600)

        assert "testing" in summary["FINAL_TEXT"].lower()


class TestArkAsrLongAudio:
    def test_should_transcribe_many_words_across_chunks(self):
        summary = _transcribe_fixture("long", timeout=1200)

        assert len(_WORD_RE.findall(summary["FINAL_TEXT"].lower())) > 100
