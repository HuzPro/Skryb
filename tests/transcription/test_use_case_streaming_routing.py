import numpy as np

from voicetotext.core.events import (
    RecordingStartedEvent,
    RecordingStoppedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.live_streaming_controller import (
    LiveStreamingController,
)
from voicetotext.transcription.streaming.streaming_engine import StreamingUpdate
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase

from tests.fakes import (
    FakeAudioRecorder,
    FakeTextOutputHandler,
    FakeTranscriptionEngine,
    InlineActionDispatcher,
    RecordingEventBus,
)


class _StubController:
    """Records routing calls; simulates worker completion in ``stop``."""

    def __init__(self, has_engine: bool = True, complete_error=None) -> None:
        self._has_engine = has_engine
        self._complete_error = complete_error
        self.start_calls = 0
        self.stop_calls = 0
        self.set_engine_calls: list = []
        self._on_complete = None
        self._active = False

    def has_engine(self) -> bool:
        return self._has_engine

    def is_active(self) -> bool:
        return self._active

    def set_engine(self, engine) -> None:
        self.set_engine_calls.append(engine)
        self._has_engine = engine is not None

    def start(self, on_complete) -> None:
        self.start_calls += 1
        self._on_complete = on_complete
        self._active = True

    def stop(self) -> None:
        self.stop_calls += 1
        self._active = False
        if self._on_complete is not None:
            self._on_complete(self._complete_error)


class _FakeStreamingEngine:
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


def _make(controller, streaming: bool = True):
    bus = RecordingEventBus()
    state_machine = AppStateMachine(bus)
    if state_machine.state is not AppState.IDLE:
        state_machine.transition_to(AppState.IDLE)
    recorder = FakeAudioRecorder()
    batch_engine = FakeTranscriptionEngine()
    use_case = RecordAndTranscribeUseCase(
        recorder=recorder,
        engine=batch_engine,
        output=FakeTextOutputHandler(),
        state_machine=state_machine,
        event_bus=bus,
        action_dispatcher=InlineActionDispatcher(),
        streaming_controller=controller,
        is_streaming_mode=lambda: streaming,
    )
    return use_case, recorder, batch_engine, state_machine, bus


class TestStreamingRouting:
    def test_should_start_streaming_not_batch_on_press(self):
        controller = _StubController()
        use_case, recorder, batch, state_machine, bus = _make(controller)
        starts: list = []
        bus.subscribe(RecordingStartedEvent, starts.append)

        use_case.on_hotkey_activated()

        assert controller.start_calls == 1
        assert recorder.start_calls == 0
        assert state_machine.state is AppState.RECORDING
        assert len(starts) == 1

    def test_should_stop_and_return_to_idle_on_second_press(self):
        controller = _StubController()
        use_case, recorder, batch, state_machine, bus = _make(controller)
        stops: list = []
        bus.subscribe(RecordingStoppedEvent, stops.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert controller.stop_calls == 1
        assert state_machine.state is AppState.IDLE
        assert batch.transcribe_calls == []
        assert len(stops) == 1

    def test_should_stop_streaming_on_deactivate(self):
        controller = _StubController()
        use_case, recorder, batch, state_machine, bus = _make(controller)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_deactivated()

        assert controller.stop_calls == 1

    def test_should_use_batch_in_batch_mode(self):
        controller = _StubController(has_engine=True)
        use_case, recorder, batch, state_machine, bus = _make(
            controller, streaming=False,
        )

        use_case.on_hotkey_activated()

        assert recorder.start_calls == 1
        assert controller.start_calls == 0

    def test_should_fall_back_to_batch_when_no_streaming_engine(self):
        controller = _StubController(has_engine=False)
        use_case, recorder, batch, state_machine, bus = _make(controller)

        use_case.on_hotkey_activated()

        assert recorder.start_calls == 1
        assert controller.start_calls == 0

    def test_should_report_failure_when_stream_completes_with_error(self):
        controller = _StubController(complete_error=RuntimeError("boom"))
        use_case, recorder, batch, state_machine, bus = _make(controller)
        failures: list = []
        bus.subscribe(TranscriptionFailedEvent, failures.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert len(failures) == 1
        assert state_machine.state is AppState.IDLE

    def test_should_recover_streaming_state_when_worker_fails_mid_run(self):
        # The worker can fail mid-run (e.g. a decode error) and call
        # on_complete(error) WITHOUT going through the user-stop path. That
        # must not leave _streaming_active latched, or the next capture's
        # stop would wrongly route to _stop_streaming and drop the audio.
        mode = {"streaming": True}
        bus = RecordingEventBus()
        state_machine = AppStateMachine(bus)
        if state_machine.state is not AppState.IDLE:
            state_machine.transition_to(AppState.IDLE)
        recorder = FakeAudioRecorder()
        recorder.will_return(
            AudioSegment(
                data=np.zeros(16000, dtype=np.float32), sample_rate=16000,
            )
        )
        batch = FakeTranscriptionEngine()
        controller = _StubController()
        use_case = RecordAndTranscribeUseCase(
            recorder=recorder,
            engine=batch,
            output=FakeTextOutputHandler(),
            state_machine=state_machine,
            event_bus=bus,
            action_dispatcher=InlineActionDispatcher(),
            streaming_controller=controller,
            is_streaming_mode=lambda: mode["streaming"],
        )
        failures: list = []
        bus.subscribe(TranscriptionFailedEvent, failures.append)

        use_case.on_hotkey_activated()  # streaming starts
        # Worker dies mid-run and reports the error out-of-band.
        controller._on_complete(RuntimeError("decode failed"))

        assert len(failures) == 1
        assert state_machine.state is AppState.IDLE

        # The app must be usable again: a batch capture now records and
        # transcribes through the batch engine. If _streaming_active were
        # left latched, the stop would misroute to _stop_streaming
        # (controller.stop) and the audio would never reach the engine.
        mode["streaming"] = False
        use_case.on_hotkey_activated()   # batch recording starts
        use_case.on_hotkey_activated()   # stop -> transcribe

        assert recorder.start_calls == 1
        assert controller.stop_calls == 0
        assert len(batch.transcribe_calls) == 1
        assert state_machine.state is AppState.IDLE


class _ScriptedSession:
    def __init__(self, committed: list[str], final: str) -> None:
        self._committed = committed
        self._final = final
        self._n = 0

    def feed(self, chunk):
        self._n += 1
        committed = self._committed[min(self._n - 1, len(self._committed) - 1)]
        return StreamingUpdate(
            committed_text=committed, tentative_text="", is_final=False,
        )

    def finalize(self):
        return TranscriptionResult(
            text=self._final, language="en",
            language_probability=0.0, transcription_seconds=0.0,
        )


class _ScriptedStreamingEngine:
    def __init__(self, session) -> None:
        self._session = session

    def load_model(self) -> None: ...

    def is_model_loaded(self) -> bool:
        return True

    def start_session(self):
        return self._session

    def dispose(self) -> None: ...


class TestRealControllerIntegration:
    def test_full_chain_types_committed_text_to_output(self):
        bus = RecordingEventBus()
        state_machine = AppStateMachine(bus)
        if state_machine.state is not AppState.IDLE:
            state_machine.transition_to(AppState.IDLE)
        recorder = FakeAudioRecorder()
        output = FakeTextOutputHandler()
        controller = LiveStreamingController(recorder, output)
        controller.set_engine(
            _ScriptedStreamingEngine(
                _ScriptedSession(["", "hello", "hello world"], "hello world"),
            )
        )
        use_case = RecordAndTranscribeUseCase(
            recorder=recorder,
            engine=FakeTranscriptionEngine(),
            output=output,
            state_machine=state_machine,
            event_bus=bus,
            action_dispatcher=InlineActionDispatcher(),
            streaming_controller=controller,
            is_streaming_mode=lambda: True,
        )

        use_case.on_hotkey_activated()
        for _ in range(3):
            recorder.captured_on_frame(np.zeros(10, dtype=np.float32))
        use_case.on_hotkey_activated()

        assert "".join(output.outputs) == "hello world"
        assert state_machine.state is AppState.IDLE


class TestEngineSlots:
    def test_set_streaming_engine_installs_and_parks_batch(self):
        controller = _StubController(has_engine=False)
        use_case, recorder, batch, state_machine, bus = _make(controller)
        streaming = _FakeStreamingEngine()

        use_case.set_streaming_engine(streaming)

        assert controller.set_engine_calls[-1] is streaming
        assert batch.dispose_called is True

    def test_set_engine_clears_the_streaming_slot(self):
        controller = _StubController(has_engine=True)
        use_case, recorder, batch, state_machine, bus = _make(controller)

        use_case.set_engine(FakeTranscriptionEngine())

        assert controller.set_engine_calls[-1] is None

    def test_dispose_disposes_both_slots(self):
        controller = _StubController(has_engine=True)
        use_case, recorder, batch, state_machine, bus = _make(controller)

        use_case.dispose()

        assert batch.dispose_called is True
        assert controller.set_engine_calls[-1] is None
