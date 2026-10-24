"""Tests for :class:`PlaceholderEngine`.

The placeholder stands in for the real transcription engine during
the LOADING phase at startup: the use case holds a reference to *an*
engine for construction, but anything that tries to actually use it
must fail loudly rather than silently produce empty output.
"""

import pytest

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.placeholder_engine import PlaceholderEngine

import numpy as np


def _make_audio() -> AudioSegment:
    return AudioSegment(
        data=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
    )


class TestPlaceholderEngineContract:
    def test_should_implement_transcription_engine_interface(self) -> None:
        assert isinstance(PlaceholderEngine(), TranscriptionEngine)


class TestPlaceholderEngineBehaviour:
    def test_should_report_not_loaded(self) -> None:
        # The placeholder is never "loaded" - it exists purely to give
        # the use case a non-None engine reference while the real one
        # is still being constructed on a background thread.
        assert PlaceholderEngine().is_model_loaded() is False

    def test_should_raise_when_transcribe_invoked(self) -> None:
        engine = PlaceholderEngine()

        with pytest.raises(TranscriptionError, match="loading"):
            engine.transcribe(_make_audio())

    def test_should_no_op_on_load_model(self) -> None:
        # ``load_model`` exists on the contract; the placeholder
        # accepts the call so that hot-swap code paths can call it
        # uniformly without inspecting the concrete type.
        PlaceholderEngine().load_model()

    def test_should_no_op_on_dispose(self) -> None:
        PlaceholderEngine().dispose()

    def test_should_no_op_on_warmup(self) -> None:
        # The default ``warmup`` calls ``transcribe`` which would
        # raise - placeholder must short-circuit so bootstrap can
        # safely warm the use case's engine regardless of which one
        # is currently attached.
        PlaceholderEngine().warmup()
