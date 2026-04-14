import logging
import threading

from voicetotext.core.action_dispatcher import ActionDispatcher
from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    ErrorOccurredEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.exceptions import EngineSwapError
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.output.handler import TextOutputHandler
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.recorder import AudioRecorder
from voicetotext.transcription.engine import TranscriptionEngine


_logger = logging.getLogger(__name__)
_MIN_DURATION_SECONDS = 0.1


class RecordAndTranscribeUseCase:

    def __init__(
        self,
        recorder: AudioRecorder,
        engine: TranscriptionEngine,
        output: TextOutputHandler,
        state_machine: AppStateMachine,
        event_bus: EventBus,
        action_dispatcher: ActionDispatcher,
    ) -> None:
        self._recorder = recorder
        self._engine = engine
        self._output = output
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._dispatcher = action_dispatcher
        self._action_lock = threading.Lock()

    def on_hotkey_activated(self) -> None:
        self._dispatcher.dispatch(self._handle_activate)

    def on_hotkey_deactivated(self) -> None:
        self._dispatcher.dispatch(self._handle_deactivate)

    def _handle_activate(self) -> None:
        with self._action_lock:
            state = self._state_machine.state
            if state == AppState.IDLE:
                self._start_recording()
            elif state == AppState.RECORDING:
                self._stop_and_transcribe()
            # TRANSCRIBING: intentionally ignored — no action while engine is busy

    def _handle_deactivate(self) -> None:
        with self._action_lock:
            if self._state_machine.state == AppState.RECORDING:
                self._stop_and_transcribe()

    def dispose_current_engine(self) -> None:
        if self._engine is not None:
            self._engine.dispose()

    def set_engine(self, engine: TranscriptionEngine) -> None:
        if self._state_machine.state != AppState.IDLE:
            raise EngineSwapError("Can only swap engine when idle")
        old_engine = self._engine
        self._engine = engine
        old_engine.dispose()

    def set_output(self, output: TextOutputHandler) -> None:
        self._output = output

    def dispose(self) -> None:
        pass

    def _start_recording(self) -> None:
        try:
            self._recorder.start_recording()
        except Exception as e:
            _logger.error("Failed to start recording: %s", e)
            self._event_bus.publish(ErrorOccurredEvent(error=str(e)))
            return
        self._state_machine.transition_to(AppState.RECORDING)
        self._event_bus.publish(RecordingStartedEvent())
        _logger.info("Recording started.")

    def _stop_and_transcribe(self) -> None:
        try:
            audio = self._recorder.stop_recording()
        except Exception as e:
            _logger.error("Failed to stop recording: %s", e)
            self._state_machine.transition_to(AppState.ERROR)
            self._event_bus.publish(ErrorOccurredEvent(error=str(e)))
            self._state_machine.transition_to(AppState.IDLE)
            return
        self._state_machine.transition_to(AppState.TRANSCRIBING)
        self._event_bus.publish(RecordingStoppedEvent())
        if audio is None:
            _logger.info("Recording stopped: no audio captured.")
            self._state_machine.transition_to(AppState.IDLE)
            return
        if audio.duration_seconds < _MIN_DURATION_SECONDS:
            _logger.info("Recording stopped: too short (%.2fs).", audio.duration_seconds)
            self._state_machine.transition_to(AppState.IDLE)
            return
        _logger.info("Recording stopped: %.1fs of audio. Transcribing...", audio.duration_seconds)
        self._dispatcher.dispatch(lambda: self._do_transcription(audio))

    def _do_transcription(self, audio: AudioSegment) -> None:
        try:
            result = self._engine.transcribe(audio)
            if result.is_empty:
                _logger.info("Transcription: (empty — no speech detected)")
            else:
                _logger.info("Transcription: '%s'", result.text)
                self._output.output(result.text)
            self._state_machine.transition_to(AppState.IDLE)
            self._event_bus.publish(
                TranscriptionCompletedEvent(
                    text=result.text,
                    duration_seconds=result.duration_seconds,
                )
            )
        except Exception as e:
            _logger.error("Transcription failed: %s", e)
            self._state_machine.transition_to(AppState.ERROR)
            self._event_bus.publish(TranscriptionFailedEvent(error=str(e)))
            self._state_machine.transition_to(AppState.IDLE)
