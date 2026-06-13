import threading

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.live_streaming_controller import (
    LiveStreamingController,
)
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionEngine,
    StreamingUpdate,
)

from tests.fakes import FakeAudioRecorder, FakeTextOutputHandler


class _FakeSession:
    def __init__(
        self, committed_per_feed, final_text,
        feed_error_at=None, finalize_error=None,
    ) -> None:
        self._committed = list(committed_per_feed)
        self._final = final_text
        self._feed_error_at = feed_error_at
        self._finalize_error = finalize_error
        self._n = 0
        self.fed: list = []

    def feed(self, chunk):
        self.fed.append(chunk)
        self._n += 1
        if self._feed_error_at is not None and self._n == self._feed_error_at:
            raise RuntimeError("feed boom")
        committed = self._committed[min(self._n - 1, len(self._committed) - 1)]
        return StreamingUpdate(
            committed_text=committed, tentative_text="", is_final=False,
        )

    def finalize(self):
        if self._finalize_error is not None:
            raise self._finalize_error
        return TranscriptionResult(
            text=self._final, language="en",
            language_probability=0.0, transcription_seconds=0.0,
        )


class _FakeStreamingEngine(StreamingTranscriptionEngine):
    def __init__(self, session) -> None:
        self._session = session
        self.disposed = False

    def load_model(self) -> None: ...

    def is_model_loaded(self) -> bool:
        return True

    def start_session(self):
        return self._session

    def dispose(self) -> None:
        self.disposed = True


def _controller(session=None):
    recorder = FakeAudioRecorder()
    out = FakeTextOutputHandler()
    controller = LiveStreamingController(recorder, out)
    if session is not None:
        controller.set_engine(_FakeStreamingEngine(session))
    return controller, recorder, out


def _frame():
    return np.zeros(10, dtype=np.float32)


class TestHappyPath:
    def test_should_emit_committed_deltas_and_complete(self):
        session = _FakeSession(["", "hello", "hello world"], "hello world done")
        controller, recorder, out = _controller(session)
        completions: list = []

        controller.start(lambda result, error: completions.append((result, error)))
        for _ in range(3):
            recorder.captured_on_frame(_frame())
        controller.stop()

        assert "".join(out.outputs) == "hello world done"
        assert len(completions) == 1
        result, error = completions[0]
        assert error is None
        assert result.text == "hello world done"  # finalized transcript handed back
        assert recorder.stop_calls == 1
        assert controller.is_active() is False

    def test_should_paste_final_word_when_finalize_repunctuates_earlier_word(self):
        # End-to-end mirror of the real Parakeet bug: the live ticks commit
        # "Testing one two," then finalize re-decodes the whole utterance as
        # "Testing. one two, three." with the first word re-punctuated. The
        # new final word must still reach the output, not be dropped.
        session = _FakeSession(
            ["Testing", "Testing one", "Testing one two,"],
            "Testing. one two, three.",
        )
        controller, recorder, out = _controller(session)

        controller.start(lambda _r, _e: None)
        for _ in range(3):
            recorder.captured_on_frame(_frame())
        controller.stop()

        assert "three." in "".join(out.outputs)

    def test_is_active_between_start_and_stop(self):
        controller, recorder, out = _controller(_FakeSession(["x"], "x"))

        controller.start(lambda _r, _e: None)
        assert controller.is_active() is True
        controller.stop()

        assert controller.is_active() is False


class TestReentrancy:
    def test_double_start_is_a_noop(self):
        controller, recorder, out = _controller(_FakeSession(["x"], "x"))

        controller.start(lambda _r, _e: None)
        controller.start(lambda _r, _e: None)

        assert recorder.start_calls == 1
        controller.stop()


class TestErrors:
    def test_feed_error_completes_with_error_and_stops_recorder(self):
        session = _FakeSession(["a", "b"], "final", feed_error_at=2)
        controller, recorder, out = _controller(session)
        done = threading.Event()
        errors: list = []

        controller.start(lambda result, error: (errors.append(error), done.set()))
        recorder.captured_on_frame(_frame())
        recorder.captured_on_frame(_frame())

        assert done.wait(timeout=3)
        assert isinstance(errors[0], RuntimeError)
        assert recorder.stop_calls == 1
        assert controller.is_active() is False

    def test_finalize_error_completes_with_error(self):
        session = _FakeSession(
            ["a"], "final", finalize_error=RuntimeError("fin boom"),
        )
        controller, recorder, out = _controller(session)
        completions: list = []

        controller.start(lambda result, error: completions.append((result, error)))
        recorder.captured_on_frame(_frame())
        controller.stop()

        result, error = completions[0]
        assert result is None
        assert isinstance(error, RuntimeError)
        assert controller.is_active() is False


class TestEngineSlot:
    def test_set_engine_disposes_the_previous_engine(self):
        controller, recorder, out = _controller()
        first = _FakeStreamingEngine(_FakeSession(["x"], "x"))
        second = _FakeStreamingEngine(_FakeSession(["y"], "y"))

        controller.set_engine(first)
        controller.set_engine(second)

        assert first.disposed is True
        assert controller.has_engine() is True

    def test_set_engine_none_disposes_and_clears(self):
        controller, recorder, out = _controller()
        engine = _FakeStreamingEngine(_FakeSession(["x"], "x"))

        controller.set_engine(engine)
        controller.set_engine(None)

        assert engine.disposed is True
        assert controller.has_engine() is False

    def test_start_without_engine_raises(self):
        controller, recorder, out = _controller()

        with pytest.raises(TranscriptionError, match="no streaming engine"):
            controller.start(lambda _r, _e: None)
