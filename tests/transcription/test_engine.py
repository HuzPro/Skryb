"""Tests for :class:`TranscriptionEngine` base behaviour.

The base class is mostly an interface, but the ``warmup`` hook has a
real default implementation: run one inference pass with a short
silent buffer so the first user-triggered transcription doesn't pay
the kernel-compilation cost. Adapters can override but get correct
behaviour for free if they don't.
"""

import numpy as np

from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.result import TranscriptionResult


class _RecordingFakeEngine(TranscriptionEngine):
    """Minimal concrete engine that records every ``transcribe`` call."""

    def __init__(self, loaded: bool = True) -> None:
        self.transcribed: list[AudioSegment] = []
        self._loaded = loaded

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        self.transcribed.append(audio)
        return TranscriptionResult(
            text="",
            language="en",
            language_probability=0.0,
            transcription_seconds=0.0,
        )

    def load_model(self) -> None:
        self._loaded = True

    def is_model_loaded(self) -> bool:
        return self._loaded

    def dispose(self) -> None:
        self._loaded = False


class TestTranscriptionEngineWarmup:
    def test_should_run_one_transcribe_pass_with_silent_audio(self) -> None:
        engine = _RecordingFakeEngine(loaded=True)

        engine.warmup()

        assert len(engine.transcribed) == 1
        audio = engine.transcribed[0]
        assert audio.sample_rate == 16000
        assert audio.duration_seconds >= 1.0, (
            "warmup buffer should be long enough to exercise the full "
            "inference path, not just the encoder prelude"
        )
        assert bool(np.all(audio.data == 0.0)), (
            "warmup audio must be silent so the result is discarded "
            "without polluting any output channel"
        )

    def test_should_skip_when_model_not_loaded(self) -> None:
        # Adapters call ``warmup`` defensively even when load_model has
        # not been called yet (e.g. during shutdown races). The default
        # implementation must be a no-op in that case rather than
        # raising.
        engine = _RecordingFakeEngine(loaded=False)

        engine.warmup()

        assert engine.transcribed == []
