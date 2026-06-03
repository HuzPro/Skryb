import math

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.transcription.streaming.parakeet_streaming_engine import (
    StreamingNanoParakeetEngine,
    _ParakeetStreamingSession,
)

_SAMPLE_RATE = 16000


class _FakeResult:
    def __init__(self, timestamp: dict) -> None:
        self.timestamp = timestamp


class _RampFakeModel:
    """Stand-in Parakeet for a ramp signal (``data[i] == i``).

    Emits one word per whole absolute second — word ``wK`` spans the
    absolute interval ``[K, K+1)`` seconds. Because it reads the absolute
    position from the ramp values, its output stays stable as the window
    grows *and* after the session trims committed audio, so the streaming
    bookkeeping can be tested faithfully (no spurious duplicate words).
    """

    def __init__(self, sample_rate: int = _SAMPLE_RATE) -> None:
        self._sr = sample_rate
        self.seen_window_samples: list[int] = []

    def transcribe(self, window: np.ndarray, timestamps: bool = False):
        self.seen_window_samples.append(int(len(window)))
        abs_start = int(round(float(window[0])))
        start_sec = abs_start / self._sr
        end_sec = (abs_start + len(window)) / self._sr
        words = []
        k = math.ceil(start_sec - 1e-9)
        while k + 1 <= end_sec + 1e-9:
            words.append(
                {"word": f"w{k}", "start": k - start_sec, "end": (k + 1) - start_sec},
            )
            k += 1
        return _FakeResult({"word": words})


class _ScriptedModel:
    """Returns a pre-scripted word list per ``transcribe`` call.

    Lets a test force adversarial hypothesis shapes (e.g. a list that
    *shrinks* after words were already committed) that the ramp model
    can't produce.
    """

    def __init__(self, scripts: list[list[str]]) -> None:
        self._scripts = scripts
        self._index = 0

    def transcribe(self, window: np.ndarray, timestamps: bool = False):
        words = self._scripts[min(self._index, len(self._scripts) - 1)]
        self._index += 1
        return _FakeResult(
            {"word": [
                {"word": w, "start": float(j), "end": float(j + 1)}
                for j, w in enumerate(words)
            ]},
        )


def _ramp_chunks(seconds: float, chunk_seconds: float = 0.1, start: int = 0):
    total = int(seconds * _SAMPLE_RATE)
    data = (np.arange(total, dtype=np.float32) + start)
    step = int(chunk_seconds * _SAMPLE_RATE)
    for i in range(0, total, step):
        yield data[i:i + step]


class TestParakeetStreamingSession:
    def test_should_grow_committed_text_monotonically(self):
        session = _ParakeetStreamingSession(_RampFakeModel())

        previous = ""
        for chunk in _ramp_chunks(3.0):
            update = session.feed(chunk)
            assert update.committed_text.startswith(previous)
            previous = update.committed_text

    def test_should_finalize_with_the_full_transcript(self):
        session = _ParakeetStreamingSession(_RampFakeModel())
        for chunk in _ramp_chunks(3.0):
            session.feed(chunk)

        result = session.finalize()

        assert result.text.split() == ["w0", "w1", "w2"]

    def test_should_not_crash_when_hypothesis_shrinks_after_commit(self):
        # Commit "a","b", then the model returns a shorter list while the
        # window is over the trim threshold — committed_count > len(ends).
        model = _ScriptedModel([["a", "b"], ["a", "b"], ["a"], ["a"]])
        session = _ParakeetStreamingSession(
            model, max_window_seconds=1.0, cadence_seconds=0.5,
        )

        for chunk in _ramp_chunks(2.0):
            session.feed(chunk)  # must not raise IndexError
        result = session.finalize()

        assert "a" in result.text.split()

    def test_should_bound_the_window_by_trimming_long_audio(self):
        model = _RampFakeModel()
        session = _ParakeetStreamingSession(model, max_window_seconds=2.0)

        for chunk in _ramp_chunks(6.0):
            session.feed(chunk)
        result = session.finalize()

        # Trimming keeps every decode window well under the 6 s of audio.
        assert max(model.seen_window_samples) < int(3.0 * _SAMPLE_RATE)
        # ...and no word is dropped or duplicated across the trims.
        assert result.text.split() == [f"w{k}" for k in range(6)]


class TestStreamingNanoParakeetEngine:
    def test_should_reject_cpu_device_as_gpu_only(self):
        engine = StreamingNanoParakeetEngine(
            repo_id="r", model_dir="m", device="cpu",
            model_factory=lambda device: _RampFakeModel(),
        )

        with pytest.raises(ValueError, match="GPU-only"):
            engine.load_model()

    def test_should_raise_when_starting_session_before_load(self):
        engine = StreamingNanoParakeetEngine(
            repo_id="r", model_dir="m",
            model_factory=lambda device: _RampFakeModel(),
        )

        with pytest.raises(TranscriptionError, match="not loaded"):
            engine.start_session()

    def test_should_start_a_session_after_loading(self):
        engine = StreamingNanoParakeetEngine(
            repo_id="r", model_dir="m",
            model_factory=lambda device: _RampFakeModel(),
        )
        engine.load_model()

        session = engine.start_session()

        assert isinstance(session, _ParakeetStreamingSession)
