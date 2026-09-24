import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from voicetotext.core.action_dispatcher import ActionDispatcher
from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    DomainEvent,
    ErrorOccurredEvent,
    RecordingCancelledEvent,
    RecordingCappedEvent,
    RecordingPausedEvent,
    RecordingResumedEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    StateChangedEvent,
    TranscriptionCompletedEvent,
    TranscriptionEmptyEvent,
    TranscriptionFailedEvent,
    TranscriptionStartedEvent,
)
from voicetotext.core.exceptions import EngineSwapError
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.diagnostics.session import SessionDiagnostics
from voicetotext.output.clipboard_write_error import ClipboardWriteError
from voicetotext.output.handler import TextOutputHandler
from voicetotext.recording.audio_segment import AudioSegment, load_wav
from voicetotext.recording.recorder import AudioRecorder
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.placeholder_engine import PlaceholderEngine
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.live_streaming_controller import (
    LiveStreamingController,
)
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionEngine,
)

_logger = logging.getLogger(__name__)
_MIN_DURATION_SECONDS = 0.1
_DEFAULT_MAX_RECORDING_SECONDS = 600.0


class RecordAndTranscribeUseCase:

    def __init__(
        self,
        recorder: AudioRecorder,
        engine: TranscriptionEngine,
        output: TextOutputHandler,
        state_machine: AppStateMachine,
        event_bus: EventBus,
        action_dispatcher: ActionDispatcher,
        diagnostics: SessionDiagnostics,
        max_recording_seconds: float = _DEFAULT_MAX_RECORDING_SECONDS,
        streaming_controller: LiveStreamingController | None = None,
        is_streaming_mode: Callable[[], bool] | None = None,
    ) -> None:
        self._recorder = recorder
        self._engine = engine
        self._output = output
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._dispatcher = action_dispatcher
        self._diagnostics = diagnostics
        self._max_recording_seconds = max_recording_seconds
        self._streaming = streaming_controller
        self._is_streaming_mode = is_streaming_mode or (lambda: False)
        self._streaming_active = False
        self._stream_requested_at: datetime | None = None
        self._action_lock = threading.Lock()
        # A retry from history runs beside live dictation; one model must
        # never decode two recordings at once.
        self._engine_lock = threading.Lock()
        # Latched signal: did the user press the hotkey while the
        # engine was still loading? On ``LOADING → IDLE`` we replay
        # exactly one activation so the user's first hotkey is honoured
        # rather than swallowed. Multiple presses during LOADING
        # coalesce to a single replay - the user's intent is "start
        # the moment you're ready", not "start N times".
        self._queued_activation = False
        # ``threading.Timer`` is one-shot: scheduled at recording start
        # to enforce the duration cap, cancelled when the user stops
        # normally or replaced when the recording finishes auto-stopped.
        self._cap_timer: threading.Timer | None = None
        # Paused time does not count towards the cap: the cap bounds the
        # audio, and a pause records none.
        self._paused = False
        self._cap_remaining = max_recording_seconds
        self._cap_started_at = 0.0
        event_bus.subscribe(StateChangedEvent, self._on_state_changed)

    def on_hotkey_activated(self) -> None:
        self._dispatcher.dispatch(self._handle_activate)

    def on_hotkey_deactivated(self) -> None:
        self._dispatcher.dispatch(self._handle_deactivate)

    def on_pause_toggled(self) -> None:
        self._dispatcher.dispatch(self._handle_pause_toggle)

    def on_cancel(self) -> None:
        self._dispatcher.dispatch(self._handle_cancel)

    def is_recording(self) -> bool:
        return self._state_machine.state == AppState.RECORDING

    def retranscribe(
        self, wav_path: Path, on_done: Callable[[str | Exception], None],
    ) -> None:
        """Transcribe a saved recording again, for history; nothing is pasted."""

        def run() -> None:
            try:
                audio = load_wav(wav_path)
                with self._engine_lock:
                    result = self._engine.transcribe(audio)
            except Exception as exc:  # noqa: BLE001 - reported to the window
                on_done(exc)
                return
            on_done(result.text)

        self._dispatcher.dispatch(run)

    def _handle_pause_toggle(self) -> None:
        with self._action_lock:
            if self._state_machine.state != AppState.RECORDING:
                return
            self._paused = not self._paused
            self._recorder.set_paused(self._paused)
            if self._paused:
                self._cap_remaining -= time.monotonic() - self._cap_started_at
                self._cancel_cap_timer()
                self._event_bus.publish(RecordingPausedEvent())
                _logger.info("Recording paused.")
            else:
                self._schedule_cap_timer()
                self._event_bus.publish(RecordingResumedEvent())
                _logger.info("Recording resumed.")

    def _handle_cancel(self) -> None:
        with self._action_lock:
            if self._state_machine.state != AppState.RECORDING:
                return
            if self._streaming_active:
                # Streaming has already typed what it heard; there is
                # nothing left to discard, so cancel just stops.
                self._stop_streaming()
                return
            self._cancel_cap_timer()
            self._paused = False
            try:
                audio = self._recorder.stop_recording()
            except Exception as exc:
                self._report_error(
                    "Failed to stop recording",
                    exc,
                    lambda message: ErrorOccurredEvent(error=message),
                )
                return
            self._state_machine.transition_to(AppState.IDLE)
            self._event_bus.publish(RecordingCancelledEvent())
            _logger.info("Recording cancelled.")
            self._dispatcher.dispatch(
                lambda: self._diagnostics.recording_cancelled(audio),
            )

    def _handle_activate(self) -> None:
        with self._action_lock:
            state = self._state_machine.state
            if state == AppState.LOADING:
                self._queued_activation = True
                _logger.info("Hotkey queued - engine still loading.")
                return
            if state == AppState.IDLE:
                self._begin_capture()
            elif state == AppState.RECORDING:
                self._end_capture()
            # TRANSCRIBING is intentionally ignored; no action while engine is busy.

    def _on_state_changed(self, event: StateChangedEvent) -> None:
        if event.old_state != AppState.LOADING.value:
            return
        if event.new_state != AppState.IDLE.value:
            return
        with self._action_lock:
            if not self._queued_activation:
                return
            self._queued_activation = False
        _logger.info("Engine ready - firing queued hotkey.")
        self._dispatcher.dispatch(self._handle_activate)

    def _handle_deactivate(self) -> None:
        with self._action_lock:
            if self._state_machine.state == AppState.RECORDING:
                self._end_capture()

    def dispose_current_engine(self) -> None:
        self._engine.dispose()

    def set_engine(self, engine: TranscriptionEngine) -> None:
        # Permit ``LOADING`` so bootstrap can attach the real engine to
        # the placeholder that the use case was constructed with on a
        # background thread, before the state machine flips to
        # ``IDLE``.
        self._require_swappable_state()
        old_engine = self._engine
        self._engine = engine
        old_engine.dispose()
        if self._streaming is not None:
            self._streaming.set_engine(None)

    def set_streaming_engine(
        self, engine: StreamingTranscriptionEngine,
    ) -> None:
        """Install the streaming engine and park the batch slot.

        The batch engine is replaced with a :class:`PlaceholderEngine` so
        its model frees its VRAM while streaming mode is active.
        """
        self._require_swappable_state()
        if self._streaming is None:
            engine.dispose()
            raise EngineSwapError("no streaming controller configured")
        self._streaming.set_engine(engine)
        old_engine = self._engine
        self._engine = PlaceholderEngine()
        old_engine.dispose()

    def _require_swappable_state(self) -> None:
        if self._state_machine.state not in {AppState.IDLE, AppState.LOADING}:
            raise EngineSwapError("Can only swap engine when idle or loading")

    def set_output(self, output: TextOutputHandler) -> None:
        self._output = output
        if self._streaming is not None:
            self._streaming.set_output(output)

    def dispose(self) -> None:
        self._diagnostics.dispose()
        self.dispose_current_engine()
        if self._streaming is not None:
            self._streaming.set_engine(None)

    def _begin_capture(self) -> None:
        if self._streaming_ready():
            self._start_streaming()
        else:
            self._start_recording()

    def _end_capture(self) -> None:
        if self._streaming_active:
            self._stop_streaming()
        else:
            self._stop_and_transcribe()

    def _streaming_ready(self) -> bool:
        return (
            self._streaming is not None
            and self._streaming.has_engine()
            and self._is_streaming_mode()
        )

    def _start_recording(self) -> None:
        self._paused = False
        self._cap_remaining = self._max_recording_seconds
        try:
            self._recorder.start_recording()
        except Exception as e:
            _logger.error("Failed to start recording: %s", e)
            self._event_bus.publish(ErrorOccurredEvent(error=str(e)))
            return
        self._state_machine.transition_to(AppState.RECORDING)
        self._event_bus.publish(RecordingStartedEvent())
        self._schedule_cap_timer()
        _logger.info("Recording started.")

    def _start_streaming(self) -> None:
        try:
            self._streaming.start(self._on_stream_complete)
        except Exception as exc:  # noqa: BLE001 - surface to the user
            _logger.error("Failed to start streaming: %s", exc)
            self._event_bus.publish(ErrorOccurredEvent(error=str(exc)))
            return
        self._paused = False
        self._cap_remaining = self._max_recording_seconds
        self._streaming_active = True
        self._state_machine.transition_to(AppState.RECORDING)
        self._event_bus.publish(RecordingStartedEvent())
        self._schedule_cap_timer()
        _logger.info("Streaming started.")

    def _stop_streaming(self) -> None:
        self._stream_requested_at = datetime.now(UTC)
        self._finish_streaming()
        self._state_machine.transition_to(AppState.TRANSCRIBING)
        self._event_bus.publish(RecordingStoppedEvent())
        _logger.info("Streaming stopped. Finalizing transcript...")
        # Finalize (a GPU decode of the trailing window) off the action
        # lock, mirroring the batch path's dispatched transcription.
        self._dispatcher.dispatch(self._streaming.stop)

    def _finish_streaming(self) -> None:
        # Tear down the per-run streaming bookkeeping. Idempotent on
        # purpose: the user-stop path runs it, and ``_on_stream_complete``
        # runs it again so a mid-run worker failure - which never passes
        # through ``_stop_streaming`` - cannot leave ``_streaming_active``
        # latched (which would misroute the next stop) or a cap timer
        # pending (which would auto-stop the next run).
        self._cancel_cap_timer()
        self._streaming_active = False

    def _on_stream_complete(
        self,
        result: TranscriptionResult | None,
        error: Exception | None,
        audio: AudioSegment | None,
    ) -> None:
        self._finish_streaming()
        requested_at = self._stream_requested_at or datetime.now(UTC)
        self._stream_requested_at = None
        if error is not None:
            self._diagnostics.transcription_finished(
                audio, error, requested_at,
            )
            self._report_error(
                "Streaming failed",
                error,
                lambda message: TranscriptionFailedEvent(error=message),
            )
            return
        if result is None:
            self._state_machine.transition_to(AppState.IDLE)
            return
        # The committed text was already pasted live by the streaming
        # writer; this only mirrors the batch path's terminal log and
        # completion event so streaming has the same observability.
        self._diagnostics.transcription_finished(
            audio, result, requested_at,
        )
        self._announce_streaming_empty(result)
        self._state_machine.transition_to(AppState.IDLE)
        self._event_bus.publish(
            TranscriptionCompletedEvent(
                text=result.text,
                transcription_seconds=result.transcription_seconds,
            )
        )

    def _announce_streaming_empty(self, result: TranscriptionResult) -> None:
        if result.is_empty:
            self._announce_empty_transcription(result)

    def _schedule_cap_timer(self) -> None:
        self._cap_started_at = time.monotonic()
        self._cap_timer = threading.Timer(
            max(self._cap_remaining, 0.0),
            self._on_cap_timer_fired,
        )
        self._cap_timer.daemon = True
        self._cap_timer.start()

    def _cancel_cap_timer(self) -> None:
        if self._cap_timer is None:
            return
        self._cap_timer.cancel()
        self._cap_timer = None

    def _on_cap_timer_fired(self) -> None:
        # Timer thread → worker thread, so the auto-stop takes the
        # same action lock the hotkey path uses. Without the
        # dispatcher hop, a cap firing mid-stop would race the lock.
        self._dispatcher.dispatch(self._handle_auto_stop)

    def _handle_auto_stop(self) -> None:
        with self._action_lock:
            if self._state_machine.state != AppState.RECORDING:
                # Cap raced a normal stop; the user already released
                # the hotkey or the state moved on. Nothing to do.
                return
            _logger.info(
                "Recording auto-stopped: %.0fs cap reached.",
                self._max_recording_seconds,
            )
            self._event_bus.publish(
                RecordingCappedEvent(
                    max_duration_seconds=self._max_recording_seconds,
                )
            )
            self._end_capture()

    def _stop_and_transcribe(self) -> None:
        requested_at = datetime.now(UTC)
        self._cancel_cap_timer()
        self._paused = False
        try:
            audio = self._recorder.stop_recording()
        except Exception as exc:
            self._report_error(
                "Failed to stop recording",
                exc,
                lambda message: ErrorOccurredEvent(error=message),
            )
            return
        self._state_machine.transition_to(AppState.TRANSCRIBING)
        self._event_bus.publish(RecordingStoppedEvent())
        if self._audio_is_skippable(audio):
            self._state_machine.transition_to(AppState.IDLE)
            return
        _logger.info(
            "Recording stopped: %.1fs of audio. Transcribing...",
            audio.duration_seconds,
        )
        self._event_bus.publish(
            TranscriptionStartedEvent(
                audio_seconds=audio.duration_seconds,
                estimated_seconds=self._diagnostics.estimate_transcription_seconds(
                    audio.duration_seconds,
                ),
            )
        )
        self._dispatcher.dispatch(
            lambda: self._do_transcription(audio, requested_at),
        )

    def _audio_is_skippable(self, audio: AudioSegment | None) -> bool:
        if audio is None:
            _logger.info("Recording stopped: no audio captured.")
            return True
        if audio.duration_seconds < _MIN_DURATION_SECONDS:
            _logger.info(
                "Recording stopped: too short (%.2fs).", audio.duration_seconds
            )
            return True
        return False

    def _do_transcription(
        self, audio: AudioSegment, requested_at: datetime,
    ) -> None:
        try:
            self._transcribe_and_publish(audio, requested_at)
        except Exception as exc:
            self._diagnostics.transcription_finished(
                audio, exc, requested_at,
            )
            self._report_error(
                "Transcription failed",
                exc,
                lambda message: TranscriptionFailedEvent(error=message),
            )

    def _transcribe_and_publish(
        self, audio: AudioSegment, requested_at: datetime,
    ) -> None:
        with self._engine_lock:
            result = self._engine.transcribe(audio)
        self._deliver_transcribed_text(result)
        self._diagnostics.transcription_finished(
            audio, result, requested_at,
        )
        self._state_machine.transition_to(AppState.IDLE)
        _logger.debug("post-transcribe: state -> IDLE")
        self._event_bus.publish(
            TranscriptionCompletedEvent(
                text=result.text,
                transcription_seconds=result.transcription_seconds,
            )
        )
        _logger.debug("post-transcribe: completion event published")

    def _deliver_transcribed_text(self, result: TranscriptionResult) -> None:
        if result.is_empty:
            self._announce_empty_transcription(result)
            return
        _logger.debug("output.output: calling")
        try:
            self._output.output(result.text)
        except ClipboardWriteError as exc:
            self._handle_clipboard_write_failure(exc)
        _logger.debug("output.output: returned")

    def _announce_empty_transcription(self, result: TranscriptionResult) -> None:
        self._event_bus.publish(
            TranscriptionEmptyEvent(
                transcription_seconds=result.transcription_seconds,
            )
        )

    def _handle_clipboard_write_failure(self, error: ClipboardWriteError) -> None:
        _logger.warning("Clipboard write failed: %s", error)
        self._event_bus.publish(ErrorOccurredEvent(error=str(error)))

    def _report_error(
        self,
        operation: str,
        error: Exception,
        make_event: Callable[[str], DomainEvent],
    ) -> None:
        _logger.error("%s: %s", operation, error)
        self._state_machine.transition_to(AppState.ERROR)
        self._event_bus.publish(make_event(str(error)))
        self._state_machine.transition_to(AppState.IDLE)
