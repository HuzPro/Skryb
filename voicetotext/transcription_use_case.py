import logging
import threading
from collections.abc import Callable

from voicetotext.core.action_dispatcher import ActionDispatcher
from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    DomainEvent,
    ErrorOccurredEvent,
    RecordingCappedEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    StateChangedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.exceptions import EngineSwapError
from voicetotext.output.clipboard_write_error import ClipboardWriteError
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.output.handler import TextOutputHandler
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.recorder import AudioRecorder
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult


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
        max_recording_seconds: float = _DEFAULT_MAX_RECORDING_SECONDS,
    ) -> None:
        self._recorder = recorder
        self._engine = engine
        self._output = output
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._dispatcher = action_dispatcher
        self._max_recording_seconds = max_recording_seconds
        self._action_lock = threading.Lock()
        # Latched signal: did the user press the hotkey while the
        # engine was still loading? On ``LOADING → IDLE`` we replay
        # exactly one activation so the user's first hotkey is honoured
        # rather than swallowed. Multiple presses during LOADING
        # coalesce to a single replay — the user's intent is "start
        # the moment you're ready", not "start N times".
        self._queued_activation = False
        # ``threading.Timer`` is one-shot: scheduled at recording start
        # to enforce the duration cap, cancelled when the user stops
        # normally or replaced when the recording finishes auto-stopped.
        self._cap_timer: threading.Timer | None = None
        event_bus.subscribe(StateChangedEvent, self._on_state_changed)

    def on_hotkey_activated(self) -> None:
        self._dispatcher.dispatch(self._handle_activate)

    def on_hotkey_deactivated(self) -> None:
        self._dispatcher.dispatch(self._handle_deactivate)

    def _handle_activate(self) -> None:
        with self._action_lock:
            state = self._state_machine.state
            if state == AppState.LOADING:
                self._queued_activation = True
                _logger.info("Hotkey queued — engine still loading.")
                return
            if state == AppState.IDLE:
                self._start_recording()
            elif state == AppState.RECORDING:
                self._stop_and_transcribe()
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
        _logger.info("Engine ready — firing queued hotkey.")
        self._dispatcher.dispatch(self._handle_activate)

    def _handle_deactivate(self) -> None:
        with self._action_lock:
            if self._state_machine.state == AppState.RECORDING:
                self._stop_and_transcribe()

    def dispose_current_engine(self) -> None:
        self._engine.dispose()

    def set_engine(self, engine: TranscriptionEngine) -> None:
        # Permit ``LOADING`` so bootstrap can attach the real engine to
        # the placeholder that the use case was constructed with on a
        # background thread, before the state machine flips to
        # ``IDLE``.
        if self._state_machine.state not in {AppState.IDLE, AppState.LOADING}:
            raise EngineSwapError("Can only swap engine when idle or loading")
        old_engine = self._engine
        self._engine = engine
        old_engine.dispose()

    def set_output(self, output: TextOutputHandler) -> None:
        self._output = output

    def dispose(self) -> None:
        self.dispose_current_engine()

    def _start_recording(self) -> None:
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

    def _schedule_cap_timer(self) -> None:
        self._cap_timer = threading.Timer(
            self._max_recording_seconds,
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
            self._stop_and_transcribe()

    def _stop_and_transcribe(self) -> None:
        self._cancel_cap_timer()
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
        self._dispatcher.dispatch(lambda: self._do_transcription(audio))

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

    def _do_transcription(self, audio: AudioSegment) -> None:
        try:
            self._transcribe_and_publish(audio)
        except Exception as exc:
            self._report_error(
                "Transcription failed",
                exc,
                lambda message: TranscriptionFailedEvent(error=message),
            )

    def _transcribe_and_publish(self, audio: AudioSegment) -> None:
        result = self._engine.transcribe(audio)
        self._deliver_transcribed_text(result)
        self._state_machine.transition_to(AppState.IDLE)
        _logger.debug("post-transcribe: state -> IDLE")
        self._event_bus.publish(
            TranscriptionCompletedEvent(
                text=result.text,
                duration_seconds=result.duration_seconds,
            )
        )
        _logger.debug("post-transcribe: completion event published")

    def _deliver_transcribed_text(self, result: TranscriptionResult) -> None:
        if result.is_empty:
            _logger.info("Transcription: (empty, no speech detected)")
            return
        _logger.info("Transcription: '%s'", result.text)
        _logger.debug("output.output: calling")
        try:
            self._output.output(result.text)
        except ClipboardWriteError as exc:
            self._handle_clipboard_write_failure(exc)
        _logger.debug("output.output: returned")

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
