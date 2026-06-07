"""Behaviour of the streaming-vs-batch comparison session decorator.

The decorator must leave live dictation byte-for-byte unchanged (pure
delegation to the wrapped session) and, at finalize, run a batch reference
engine over the FULL captured audio and log the two transcripts side by side.
The batch pass is best-effort: a failure is logged, never raised, and the real
streaming result is always returned.
"""
from __future__ import annotations

import numpy as np

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionSession, StreamingUpdate,
)
from voicetotext.transcription.streaming.comparing_streaming_session import (
    ComparingStreamingSession,
)

_UPDATE = StreamingUpdate(committed_text="hello", tentative_text="", is_final=False)
_STREAMING_RESULT = TranscriptionResult(
    text="hello world", language="en",
    language_probability=0.0, transcription_seconds=1.0,
)
_BATCH_RESULT = TranscriptionResult(
    text="Hello, world.", language="en",
    language_probability=0.0, transcription_seconds=2.0,
)


class _FakeInnerSession(StreamingTranscriptionSession):
    def __init__(
        self, update: StreamingUpdate, result: TranscriptionResult,
    ) -> None:
        self._update = update
        self._result = result
        self.fed: list[np.ndarray] = []

    def feed(self, chunk: np.ndarray) -> StreamingUpdate:
        self.fed.append(chunk)
        return self._update

    def finalize(self) -> TranscriptionResult:
        return self._result


class _FakeBatchEngine:
    def __init__(
        self,
        result: TranscriptionResult | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.segments: list[AudioSegment] = []

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        self.segments.append(audio)
        if self._error is not None:
            raise self._error
        return self._result


def _chunk(*samples: float) -> np.ndarray:
    return np.array(samples, dtype=np.float32)


class TestLiveOutputIsUnchanged:
    def test_feed_delegates_to_inner_and_returns_its_update(self) -> None:
        inner = _FakeInnerSession(_UPDATE, _STREAMING_RESULT)
        session = ComparingStreamingSession(inner, _FakeBatchEngine(_BATCH_RESULT))

        update = session.feed(_chunk(1, 2))

        assert update is _UPDATE
        assert np.array_equal(inner.fed[0], _chunk(1, 2))

    def test_finalize_returns_the_inner_streaming_result(self) -> None:
        inner = _FakeInnerSession(_UPDATE, _STREAMING_RESULT)
        session = ComparingStreamingSession(inner, _FakeBatchEngine(_BATCH_RESULT))
        session.feed(_chunk(1, 2))

        result = session.finalize()

        assert result is _STREAMING_RESULT


class TestBatchComparison:
    def test_runs_batch_once_over_all_captured_audio(self) -> None:
        inner = _FakeInnerSession(_UPDATE, _STREAMING_RESULT)
        batch = _FakeBatchEngine(_BATCH_RESULT)
        session = ComparingStreamingSession(inner, batch)
        session.feed(_chunk(1, 2))
        session.feed(_chunk(3, 4, 5))

        session.finalize()

        assert len(batch.segments) == 1
        segment = batch.segments[0]
        assert segment.sample_rate == 16000
        assert np.array_equal(segment.data, _chunk(1, 2, 3, 4, 5))

    def test_logs_both_streaming_and_batch_transcripts(self, app_caplog) -> None:
        inner = _FakeInnerSession(_UPDATE, _STREAMING_RESULT)
        session = ComparingStreamingSession(inner, _FakeBatchEngine(_BATCH_RESULT))
        session.feed(_chunk(1, 2))

        session.finalize()

        assert "hello world" in app_caplog.text       # streaming transcript
        assert "Hello, world." in app_caplog.text      # batch transcript

    def test_comparison_log_includes_the_streaming_duration(self, app_caplog) -> None:
        inner = _FakeInnerSession(_UPDATE, _STREAMING_RESULT)  # 1.0s
        session = ComparingStreamingSession(inner, _FakeBatchEngine(_BATCH_RESULT))
        session.feed(_chunk(1, 2))

        session.finalize()

        assert "1.00s" in app_caplog.text   # timing folded into the side-by-side line

    def test_batch_failure_is_logged_and_does_not_break_finalize(
        self, app_caplog,
    ) -> None:
        inner = _FakeInnerSession(_UPDATE, _STREAMING_RESULT)
        exploding = _FakeBatchEngine(error=RuntimeError("kaboom"))
        session = ComparingStreamingSession(inner, exploding)
        session.feed(_chunk(1, 2))

        result = session.finalize()

        assert result is _STREAMING_RESULT
        assert "kaboom" in app_caplog.text

    def test_no_batch_engine_skips_comparison(self) -> None:
        inner = _FakeInnerSession(_UPDATE, _STREAMING_RESULT)
        session = ComparingStreamingSession(inner, None)
        session.feed(_chunk(1, 2))

        result = session.finalize()

        assert result is _STREAMING_RESULT
