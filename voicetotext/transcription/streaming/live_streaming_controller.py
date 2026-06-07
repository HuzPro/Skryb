import logging
import queue
import threading
from collections.abc import Callable

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.output.handler import TextOutputHandler
from voicetotext.recording.recorder import AudioRecorder
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionEngine,
)
from voicetotext.transcription.streaming.streaming_output_writer import (
    StreamingOutputWriter,
)


_logger = logging.getLogger(__name__)
_SENTINEL = object()

# The worker hands back the finalized transcript on success (``error`` None)
# or the failure (``result`` None) so the use case can log it and publish a
# completion event carrying the real text and duration.
OnComplete = Callable[[TranscriptionResult | None, Exception | None], None]


class LiveStreamingController:
    """Drives one live streaming session: mic frames -> worker -> output.

    The recorder's audio callback only enqueues frames; a dedicated worker
    thread runs the GPU decode and pastes committed deltas, so the audio
    thread is never blocked. It owns only the mechanics — the use case
    keeps ownership of state transitions and events, and is told when a
    run finishes via the ``on_complete`` callback.
    """

    def __init__(
        self, recorder: AudioRecorder, output: TextOutputHandler,
    ) -> None:
        self._recorder = recorder
        self._writer = StreamingOutputWriter(output)
        self._engine: StreamingTranscriptionEngine | None = None
        self._session: object | None = None
        self._queue: queue.Queue | None = None
        self._worker: threading.Thread | None = None
        self._on_complete: OnComplete = lambda _result, _error: None
        self._active = False
        self._lock = threading.Lock()

    def set_engine(self, engine: StreamingTranscriptionEngine | None) -> None:
        old = self._engine
        self._engine = engine
        if old is not None and old is not engine:
            old.dispose()

    def set_output(self, output: TextOutputHandler) -> None:
        self._writer = StreamingOutputWriter(output)

    def has_engine(self) -> bool:
        return self._engine is not None

    def is_active(self) -> bool:
        return self._active

    def start(self, on_complete: OnComplete) -> None:
        with self._lock:
            if self._active:
                return
            if self._engine is None:
                raise TranscriptionError("no streaming engine loaded")
            self._on_complete = on_complete
            self._session = self._engine.start_session()
            self._writer.reset()
            self._queue = queue.Queue()
            self._worker = threading.Thread(
                target=self._run, name="stream-worker", daemon=True,
            )
            self._active = True
            self._worker.start()
        self._recorder.start_recording(on_frame=self._queue.put)

    def stop(self) -> None:
        """Stop capture and finalize. Blocks until the worker is done."""
        with self._lock:
            if not self._active:
                return
            worker = self._worker
            frame_queue = self._queue
        # Stop the mic BEFORE the sentinel so no late frame lands after it.
        self._recorder.stop_recording()
        frame_queue.put(_SENTINEL)
        if worker is not None:
            worker.join()

    def _run(self) -> None:
        on_complete = self._on_complete
        result: TranscriptionResult | None = None
        error: Exception | None = None
        try:
            result = self._consume_until_sentinel()
        except Exception as exc:  # noqa: BLE001 — surfaced via on_complete
            _logger.exception("streaming worker failed")
            error = exc
            self._stop_recorder_quietly()
        finally:
            self._reset()
        on_complete(result, error)

    def _consume_until_sentinel(self) -> TranscriptionResult:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            update = self._session.feed(item)
            self._writer.emit(update.committed_text)
        result = self._session.finalize()
        self._writer.emit(result.text)
        return result

    def _stop_recorder_quietly(self) -> None:
        try:
            self._recorder.stop_recording()
        except Exception:  # noqa: BLE001 — best-effort mic teardown on failure
            _logger.debug("recorder stop during cleanup failed", exc_info=True)

    def _reset(self) -> None:
        with self._lock:
            self._active = False
            self._worker = None
            self._queue = None
            self._session = None
