"""Lifecycle/wiring of the streaming-vs-batch comparison engine decorator.

The engine wraps a real streaming engine and owns a batch reference engine.
It loads BOTH, hands out comparison sessions, and disposes both. The batch
reference is optional: if it fails to load, the live streaming engine keeps
working and the comparison is silently disabled (logged).
"""
from __future__ import annotations

import numpy as np

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionEngine, StreamingTranscriptionSession, StreamingUpdate,
)
from voicetotext.transcription.streaming.comparing_streaming_session import (
    ComparingStreamingSession,
)
from voicetotext.transcription.streaming.comparing_streaming_engine import (
    ComparingStreamingEngine,
)

_UPDATE = StreamingUpdate(committed_text="hi", tentative_text="", is_final=False)
_STREAMING_RESULT = TranscriptionResult(
    text="hi there", language="en",
    language_probability=0.0, transcription_seconds=1.0,
)
_BATCH_RESULT = TranscriptionResult(
    text="Hi there.", language="en",
    language_probability=0.0, transcription_seconds=2.0,
)


class _FakeInnerSession(StreamingTranscriptionSession):
    def feed(self, chunk: np.ndarray) -> StreamingUpdate:
        return _UPDATE

    def finalize(self) -> TranscriptionResult:
        return _STREAMING_RESULT


class _FakeStreamingEngine(StreamingTranscriptionEngine):
    def __init__(self) -> None:
        self.loaded = False
        self.warmed = False
        self.disposed = False

    def load_model(self) -> None:
        self.loaded = True

    def is_model_loaded(self) -> bool:
        return self.loaded

    def warmup(self) -> None:
        self.warmed = True

    def start_session(self) -> StreamingTranscriptionSession:
        return _FakeInnerSession()

    def dispose(self) -> None:
        self.disposed = True


class _FakeBatchEngine:
    def __init__(self, *, load_error: Exception | None = None) -> None:
        self._load_error = load_error
        self.loaded = False
        self.disposed = False
        self.segments: list[AudioSegment] = []

    def load_model(self) -> None:
        if self._load_error is not None:
            raise self._load_error
        self.loaded = True

    def is_model_loaded(self) -> bool:
        return self.loaded

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        self.segments.append(audio)
        return _BATCH_RESULT

    def dispose(self) -> None:
        self.disposed = True


class TestLifecycle:
    def test_load_model_loads_both_streaming_and_batch(self) -> None:
        streaming, batch = _FakeStreamingEngine(), _FakeBatchEngine()

        ComparingStreamingEngine(streaming, batch).load_model()

        assert streaming.loaded is True
        assert batch.loaded is True

    def test_is_model_loaded_reflects_the_streaming_engine(self) -> None:
        engine = ComparingStreamingEngine(_FakeStreamingEngine(), _FakeBatchEngine())

        assert engine.is_model_loaded() is False
        engine.load_model()
        assert engine.is_model_loaded() is True

    def test_warmup_delegates_to_the_streaming_engine(self) -> None:
        streaming = _FakeStreamingEngine()

        ComparingStreamingEngine(streaming, _FakeBatchEngine()).warmup()

        assert streaming.warmed is True

    def test_dispose_disposes_both_engines(self) -> None:
        streaming, batch = _FakeStreamingEngine(), _FakeBatchEngine()
        engine = ComparingStreamingEngine(streaming, batch)
        engine.load_model()

        engine.dispose()

        assert streaming.disposed is True
        assert batch.disposed is True


class TestSession:
    def test_start_session_produces_a_comparing_session(self) -> None:
        engine = ComparingStreamingEngine(_FakeStreamingEngine(), _FakeBatchEngine())
        engine.load_model()

        session = engine.start_session()

        assert isinstance(session, ComparingStreamingSession)

    def test_session_runs_the_batch_comparison_on_finalize(self) -> None:
        batch = _FakeBatchEngine()
        engine = ComparingStreamingEngine(_FakeStreamingEngine(), batch)
        engine.load_model()

        session = engine.start_session()
        session.feed(np.ones(4, dtype=np.float32))
        session.finalize()

        assert len(batch.segments) == 1


class TestBatchReferenceDegradation:
    def test_batch_load_failure_keeps_streaming_and_disables_comparison(
        self, app_caplog,
    ) -> None:
        streaming = _FakeStreamingEngine()
        batch = _FakeBatchEngine(load_error=RuntimeError("no batch weights"))
        engine = ComparingStreamingEngine(streaming, batch)

        engine.load_model()  # must NOT raise

        assert engine.is_model_loaded() is True
        assert "no batch weights" in app_caplog.text

        session = engine.start_session()
        session.feed(np.ones(4, dtype=np.float32))
        session.finalize()
        assert batch.segments == []  # comparison disabled — batch never ran

    def test_dispose_after_failed_batch_load_skips_batch_dispose(self) -> None:
        streaming = _FakeStreamingEngine()
        batch = _FakeBatchEngine(load_error=RuntimeError("x"))
        engine = ComparingStreamingEngine(streaming, batch)
        engine.load_model()

        engine.dispose()

        assert streaming.disposed is True
        assert batch.disposed is False  # was nulled, never loaded
