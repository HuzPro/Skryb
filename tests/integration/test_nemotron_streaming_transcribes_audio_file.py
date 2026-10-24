"""Real cache-aware nemotron streaming over real audio, in a clean subprocess.

The production streaming path (``nemotron_streaming`` models ->
:class:`CacheAwareStreamingEngine` -> :func:`load_streaming_model`) is the
branch's headline feature, and ROCm compatibility on AMD is the one open
risk. This drives that exact path end to end via
``scripts/bench_cache_aware_streaming.py`` in a *fresh* interpreter -- the same
isolation the Parakeet streaming test uses, because loading the
sentencepiece-backed model alongside pytest's Qt plugin segfaults on Windows.

Opt-in (``SKRYB_RUN_REAL_MODEL_TESTS``), GPU-only, skipped when the
``.nemo`` isn't downloaded. Assertions are lenient -- streaming revises its
tail, so we pin behaviours any working run satisfies.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.helpers.gpu import require_cuda
from tests.helpers.model_cache import models_dir


pytestmark = [pytest.mark.integration, pytest.mark.slow]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _REPO_ROOT / "scripts" / "bench_cache_aware_streaming.py"
_REPO_ID = "nvidia/nemotron-speech-streaming-en-0.6b"
_NEMO_FILENAME = "nemotron-speech-streaming-en-0.6b.nemo"
_OPT_IN = "SKRYB_RUN_REAL_MODEL_TESTS"
_WORD_RE = re.compile(r"[a-z0-9]+")


def _skip_unless_nemotron_ready() -> None:
    if not os.environ.get(_OPT_IN):
        pytest.skip(f"set {_OPT_IN} to run real-model streaming tests")
    require_cuda()
    directory = models_dir()
    if directory is None:
        pytest.skip("models dir unavailable")
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(
        repo_id=_REPO_ID, filename=_NEMO_FILENAME, cache_dir=str(directory),
    )
    if not isinstance(cached, str):
        pytest.skip(f"nemotron streaming model not installed: {_REPO_ID}")


def _stream_fixture(name: str, *extra: str, timeout: float) -> dict[str, str]:
    _skip_unless_nemotron_ready()
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


class TestNemotronStreamingShortAudio:
    def test_should_transcribe_testing_one_two_three(self):
        summary = _stream_fixture("short", timeout=180)

        assert "testing" in summary["FINAL_TEXT"].lower()

    def test_should_emit_partial_updates_while_streaming(self):
        summary = _stream_fixture("short", timeout=180)

        assert int(summary["PARTIAL_COUNT"]) > 0


class TestNemotronStreamingLongAudio:
    def test_should_emit_distinct_partials_and_many_words(self):
        summary = _stream_fixture("long", timeout=600)

        assert int(summary["DISTINCT_PARTIALS"]) >= 2
        assert len(_WORD_RE.findall(summary["FINAL_TEXT"])) > 100
