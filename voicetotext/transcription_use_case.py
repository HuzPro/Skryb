from concurrent.futures import ThreadPoolExecutor, Future

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    RecordingStartedEvent,
    RecordingStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.output.handler import TextOutputHandler
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.recorder import AudioRecorder
from voicetotext.transcription.engine import TranscriptionEngine


class RecordAndTranscribeUseCase:

    def __init__(
        self,
        recorder: AudioRecorder,
        engine: TranscriptionEngine,
        output: TextOutputHandler,
        state_machine: AppStateMachine,
        event_bus: EventBus,
    ):
        self._recorder = recorder
        self._engine = engine
        self._output = output
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_future: Future | None = None

    def on_hotkey_activated(self) -> None:
        state = self._state_machine.state

        if state == AppState.IDLE:
            self._start_recording()
        elif state == AppState.RECORDING:
            self._stop_and_transcribe()

    def on_hotkey_deactivated(self) -> None:
        if self._state_machine.state == AppState.RECORDING:
            self._stop_and_transcribe()

    def set_engine(self, engine: TranscriptionEngine) -> None:
        if self._state_machine.state != AppState.IDLE:
            raise RuntimeError("Can only swap engine when idle")
        self._engine = engine

    def set_output(self, output: TextOutputHandler) -> None:
        self._output = output

    def dispose(self) -> None:
        self._executor.shutdown(wait=False)

    def _start_recording(self) -> None:
        self._state_machine.transition_to(AppState.RECORDING)
        self._recorder.start_recording()
        self._event_bus.publish(RecordingStartedEvent())

    def _stop_and_transcribe(self) -> None:
        audio = self._recorder.stop_recording()
        self._state_machine.transition_to(AppState.TRANSCRIBING)
        self._event_bus.publish(RecordingStoppedEvent())
        self._current_future = self._executor.submit(self._do_transcription, audio)

    def _do_transcription(self, audio: AudioSegment) -> None:
        try:
            result = self._engine.transcribe(audio)
            if not result.is_empty:
                self._output.output(result.text)
            self._state_machine.transition_to(AppState.IDLE)
            self._event_bus.publish(
                TranscriptionCompletedEvent(
                    text=result.text,
                    duration_seconds=result.duration_seconds,
                )
            )
        except Exception as e:
            self._state_machine.transition_to(AppState.ERROR)
            self._event_bus.publish(TranscriptionFailedEvent(error=str(e)))
            self._state_machine.transition_to(AppState.IDLE)

    def _wait_for_transcription(self) -> None:
        if self._current_future:
            self._current_future.result(timeout=10)
            self._current_future = None
