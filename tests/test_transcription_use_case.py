from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.fakes.dispatcher import DeferredActionDispatcher, InlineActionDispatcher
from voicetotext.core.action_dispatcher import ActionDispatcher
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    ErrorOccurredEvent,
    RecordingCappedEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.exceptions import AudioRecorderError, EngineSwapError
from voicetotext.output.clipboard_write_error import ClipboardWriteError
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(data=np.zeros(samples, dtype=np.float32), sample_rate=16000)


def _make_result(text: str = "hello world") -> TranscriptionResult:
    return TranscriptionResult(
        text=text, language="en", language_probability=0.95, duration_seconds=0.5
    )


def _build_use_case(
    dispatcher: ActionDispatcher | None = None,
    initial_state: AppState = AppState.IDLE,
    max_recording_seconds: float = 600.0,
):
    event_bus = SimpleEventBus()
    state_machine = AppStateMachine(event_bus, initial_state=initial_state)
    recorder = MagicMock()
    engine = MagicMock()
    output = MagicMock()
    use_case = RecordAndTranscribeUseCase(
        recorder=recorder,
        engine=engine,
        output=output,
        state_machine=state_machine,
        event_bus=event_bus,
        action_dispatcher=dispatcher or InlineActionDispatcher(),
        max_recording_seconds=max_recording_seconds,
    )
    return use_case, recorder, engine, output, event_bus, state_machine


class TestActivateFromIdle:
    def test_should_start_recording_on_activate_when_idle(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case()

        use_case.on_hotkey_activated()

        recorder.start_recording.assert_called_once()
        assert state_machine.state == AppState.RECORDING

    def test_should_publish_recording_started_event(self):
        use_case, _, _, _, event_bus, _ = _build_use_case()
        received = []
        event_bus.subscribe(RecordingStartedEvent, received.append)

        use_case.on_hotkey_activated()

        assert len(received) == 1


class TestActivateFromRecording:
    def test_should_stop_recording_and_begin_transcription(self):
        use_case, recorder, engine, _, _, _ = _build_use_case()
        audio = _make_audio()
        recorder.stop_recording.return_value = audio
        engine.transcribe.return_value = _make_result("hello")

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        recorder.stop_recording.assert_called_once()
        engine.transcribe.assert_called_once_with(audio)

    def test_should_send_text_to_output_handler(self):
        use_case, recorder, engine, output, _, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        output.output.assert_called_once_with("hello")

    def test_should_not_output_empty_transcription(self):
        use_case, recorder, engine, output, _, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("")

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        output.output.assert_not_called()

    def test_should_return_to_idle_after_transcription(self):
        use_case, recorder, engine, _, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert state_machine.state == AppState.IDLE

    def test_should_publish_recording_stopped_event(self):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")
        received = []
        event_bus.subscribe(RecordingStoppedEvent, received.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert len(received) == 1

    def test_should_publish_transcription_completed_event(self):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")
        received = []
        event_bus.subscribe(TranscriptionCompletedEvent, received.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert len(received) == 1
        assert received[0].text == "hello"


class TestActivateFromTranscribing:
    def test_should_ignore_activate_while_state_is_transcribing(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case()
        state_machine.transition_to(AppState.RECORDING)
        state_machine.transition_to(AppState.TRANSCRIBING)

        use_case.on_hotkey_activated()

        recorder.start_recording.assert_not_called()
        recorder.stop_recording.assert_not_called()
        assert state_machine.state == AppState.TRANSCRIBING


class TestDeactivateForPushToTalk:
    def test_should_stop_and_transcribe_on_deactivate(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")

        use_case.on_hotkey_activated()
        use_case.on_hotkey_deactivated()

        recorder.stop_recording.assert_called_once()
        engine.transcribe.assert_called_once()
        output.output.assert_called_once_with("hello")
        assert state_machine.state == AppState.IDLE

    def test_should_ignore_deactivate_when_not_recording(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case()

        use_case.on_hotkey_deactivated()

        recorder.stop_recording.assert_not_called()
        assert state_machine.state == AppState.IDLE


class TestTranscriptionError:
    def test_should_return_to_idle_on_transcription_failure(self):
        use_case, recorder, engine, _, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.side_effect = RuntimeError("model crashed")

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert state_machine.state == AppState.IDLE

    def test_should_publish_transcription_failed_event_on_error(self):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.side_effect = RuntimeError("model crashed")
        received = []
        event_bus.subscribe(TranscriptionFailedEvent, received.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert len(received) == 1
        assert "model crashed" in received[0].error


class TestClipboardWriteFailure:
    def test_should_publish_error_event_when_clipboard_write_fails(self):
        use_case, recorder, engine, output, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")
        output.output.side_effect = ClipboardWriteError("GlobalLock returned NULL")
        received: list[ErrorOccurredEvent] = []
        event_bus.subscribe(ErrorOccurredEvent, received.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert len(received) == 1
        assert "GlobalLock returned NULL" in received[0].error

    def test_should_return_to_idle_when_clipboard_write_fails(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")
        output.output.side_effect = ClipboardWriteError("GlobalLock returned NULL")

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert state_machine.state == AppState.IDLE


class TestHotSwapEngine:
    def test_should_swap_engine_when_idle(self):
        use_case, recorder, engine, _, _, _ = _build_use_case()
        new_engine = MagicMock()
        new_engine.transcribe.return_value = _make_result("from new engine")
        recorder.stop_recording.return_value = _make_audio()

        use_case.set_engine(new_engine)
        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        new_engine.transcribe.assert_called_once()
        engine.transcribe.assert_not_called()

    def test_should_reject_engine_swap_when_not_idle(self):
        use_case, _, _, _, _, _ = _build_use_case()
        use_case.on_hotkey_activated()

        with pytest.raises(EngineSwapError, match="idle"):
            use_case.set_engine(MagicMock())

    def test_should_dispose_old_engine_when_swapped_out(self):
        use_case, _, old_engine, _, _, _ = _build_use_case()
        new_engine = MagicMock()

        use_case.set_engine(new_engine)

        old_engine.dispose.assert_called_once()

    def test_should_not_dispose_new_engine_when_swap_rejected(self):
        use_case, _, old_engine, _, _, _ = _build_use_case()
        use_case.on_hotkey_activated()
        new_engine = MagicMock()

        with pytest.raises(EngineSwapError):
            use_case.set_engine(new_engine)

        old_engine.dispose.assert_not_called()
        new_engine.dispose.assert_not_called()


class TestLoadingStateBehaviour:
    """While the engine is loading on a background thread, the use case
    holds a :class:`PlaceholderEngine` and must (a) accept the real
    engine via ``set_engine`` and (b) ignore hotkey presses safely —
    queueing one for replay so the user's "press hotkey the moment
    the app starts" expectation works once the engine is ready.
    """

    def test_should_allow_set_engine_during_loading(self):
        use_case, _, old_engine, _, _, _ = _build_use_case(
            initial_state=AppState.LOADING,
        )
        new_engine = MagicMock()

        use_case.set_engine(new_engine)

        old_engine.dispose.assert_called_once()

    def test_should_not_start_recording_when_hotkey_pressed_during_loading(
        self,
    ):
        use_case, recorder, _, _, _, state_machine = _build_use_case(
            initial_state=AppState.LOADING,
        )

        use_case.on_hotkey_activated()

        recorder.start_recording.assert_not_called()
        assert state_machine.state == AppState.LOADING

    def test_should_replay_queued_hotkey_when_loading_completes(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case(
            initial_state=AppState.LOADING,
        )
        use_case.on_hotkey_activated()

        state_machine.transition_to(AppState.IDLE)

        recorder.start_recording.assert_called_once()
        assert state_machine.state == AppState.RECORDING

    def test_should_do_nothing_on_loading_to_idle_when_no_hotkey_queued(
        self,
    ):
        use_case, recorder, _, _, _, state_machine = _build_use_case(
            initial_state=AppState.LOADING,
        )

        state_machine.transition_to(AppState.IDLE)

        recorder.start_recording.assert_not_called()
        assert state_machine.state == AppState.IDLE

    def test_should_coalesce_repeated_hotkey_presses_during_loading(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case(
            initial_state=AppState.LOADING,
        )

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()
        state_machine.transition_to(AppState.IDLE)

        recorder.start_recording.assert_called_once()


class TestDisposeCurrentEngine:
    def test_should_dispose_current_engine(self):
        use_case, _, engine, _, _, _ = _build_use_case()

        use_case.dispose_current_engine()

        engine.dispose.assert_called_once()

    def test_should_not_raise_when_called_twice(self):
        use_case, _, engine, _, _, _ = _build_use_case()

        use_case.dispose_current_engine()
        use_case.dispose_current_engine()


class TestRecordingDurationCap:
    """A hard recording cap auto-stops the recorder after a configured
    duration so a forgotten hotkey (or held-down push-to-talk) doesn't
    fill memory with audio that no engine can actually transcribe.
    The cap is implemented as a ``threading.Timer`` scheduled at
    recording start and cancelled at recording stop.
    """

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_should_schedule_cap_timer_when_recording_starts(self, MockTimer):
        from unittest.mock import ANY

        use_case, recorder, *_ = _build_use_case(
            max_recording_seconds=42.0,
        )

        use_case.on_hotkey_activated()

        MockTimer.assert_called_once_with(42.0, ANY)
        MockTimer.return_value.start.assert_called_once()

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_should_cancel_cap_timer_when_recording_stops_normally(
        self, MockTimer,
    ):
        use_case, recorder, engine, _, _, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("done")

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        MockTimer.return_value.cancel.assert_called_once()

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_should_auto_stop_when_cap_timer_fires(self, MockTimer):
        use_case, recorder, engine, _, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("auto-capped")
        use_case.on_hotkey_activated()

        # Pull the callback the use case handed to ``Timer`` and fire
        # it directly — equivalent to the real timer thread firing.
        _interval, callback = MockTimer.call_args.args
        callback()

        recorder.stop_recording.assert_called_once()
        engine.transcribe.assert_called_once()
        assert state_machine.state == AppState.IDLE

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_should_publish_recording_capped_event_on_auto_stop(
        self, MockTimer,
    ):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case(
            max_recording_seconds=600.0,
        )
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("done")
        received: list[RecordingCappedEvent] = []
        event_bus.subscribe(RecordingCappedEvent, received.append)
        use_case.on_hotkey_activated()

        _interval, callback = MockTimer.call_args.args
        callback()

        assert len(received) == 1
        assert received[0].max_duration_seconds == 600.0

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_should_no_op_when_cap_fires_after_recording_already_stopped(
        self, MockTimer,
    ):
        # Real-world race: user releases push-to-talk a millisecond
        # before the timer thread wakes up. The callback must see
        # state != RECORDING and exit cleanly without double-stopping
        # the recorder or republishing the capped event.
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("normal")
        capped: list[RecordingCappedEvent] = []
        event_bus.subscribe(RecordingCappedEvent, capped.append)
        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()  # normal stop

        _interval, callback = MockTimer.call_args.args
        callback()  # late-firing cap; should be a no-op

        assert recorder.stop_recording.call_count == 1
        assert capped == []


class TestHotSwapOutput:
    def test_should_swap_output_handler(self):
        use_case, recorder, engine, _, _, _ = _build_use_case()
        new_output = MagicMock()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")

        use_case.set_output(new_output)
        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        new_output.output.assert_called_once_with("hello")


class TestStartRecordingFailure:
    def test_should_recover_to_idle_when_start_recording_fails(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case()
        recorder.start_recording.side_effect = AudioRecorderError("mic unavailable")

        use_case.on_hotkey_activated()

        assert state_machine.state == AppState.IDLE

    def test_should_publish_error_event_when_start_recording_fails(self):
        use_case, recorder, _, _, event_bus, _ = _build_use_case()
        recorder.start_recording.side_effect = AudioRecorderError("mic unavailable")
        received = []
        event_bus.subscribe(ErrorOccurredEvent, received.append)

        use_case.on_hotkey_activated()

        assert len(received) == 1
        assert "mic unavailable" in received[0].error


class TestStopRecordingFailure:
    def test_should_recover_to_idle_when_stop_recording_fails(self):
        use_case, recorder, _, _, _, state_machine = _build_use_case()
        recorder.stop_recording.side_effect = AudioRecorderError("stream error")

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert state_machine.state == AppState.IDLE

    def test_should_publish_error_event_when_stop_recording_fails(self):
        use_case, recorder, _, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.side_effect = AudioRecorderError("stream error")
        received = []
        event_bus.subscribe(ErrorOccurredEvent, received.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert len(received) == 1
        assert "stream error" in received[0].error


class TestShortRecording:
    def test_should_skip_transcription_for_audio_shorter_than_minimum(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        short_audio = AudioSegment(
            data=np.zeros(800, dtype=np.float32),
            sample_rate=16000,
        )
        recorder.stop_recording.return_value = short_audio

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        engine.transcribe.assert_not_called()
        output.output.assert_not_called()
        assert state_machine.state == AppState.IDLE


class TestEmptyRecording:
    def test_should_skip_transcription_when_recording_is_empty(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = None

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        engine.transcribe.assert_not_called()
        output.output.assert_not_called()
        assert state_machine.state == AppState.IDLE

    def test_should_publish_recording_stopped_event_even_when_empty(self):
        use_case, recorder, _, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = None
        received = []
        event_bus.subscribe(RecordingStoppedEvent, received.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert len(received) == 1


class TestDispatchesThroughActionDispatcher:
    """The use case must never touch the recorder or engine directly from
    the hotkey callback — everything goes through the ActionDispatcher so
    pynput's callback thread is never blocked by slow work."""

    def test_should_dispatch_activate_handler_rather_than_calling_recorder_directly(self):
        dispatcher = DeferredActionDispatcher()
        use_case, recorder, _, _, _, _ = _build_use_case(dispatcher=dispatcher)

        use_case.on_hotkey_activated()

        recorder.start_recording.assert_not_called()
        assert len(dispatcher.pending) == 1

    def test_should_dispatch_deactivate_handler_rather_than_calling_recorder_directly(self):
        dispatcher = DeferredActionDispatcher()
        use_case, recorder, _, _, _, _ = _build_use_case(dispatcher=dispatcher)

        use_case.on_hotkey_deactivated()

        recorder.stop_recording.assert_not_called()
        assert len(dispatcher.pending) == 1

    def test_should_dispatch_transcription_rather_than_blocking_handler(self):
        dispatcher = DeferredActionDispatcher()
        use_case, recorder, engine, _, _, state_machine = _build_use_case(dispatcher=dispatcher)
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")

        use_case.on_hotkey_activated()
        dispatcher.drain_next()  # runs _handle_activate → _start_recording

        use_case.on_hotkey_activated()
        dispatcher.drain_next()  # runs _handle_activate → _stop_and_transcribe → dispatches _do_transcription

        # Transcription is queued but has not run yet — state is TRANSCRIBING.
        assert state_machine.state == AppState.TRANSCRIBING
        engine.transcribe.assert_not_called()
        assert len(dispatcher.pending) == 1

        dispatcher.drain_next()  # runs _do_transcription
        assert state_machine.state == AppState.IDLE
        engine.transcribe.assert_called_once()


class TestDispose:
    def test_should_dispose_current_engine(self):
        use_case, _, engine, _, _, _ = _build_use_case()

        use_case.dispose()

        engine.dispose.assert_called_once()
