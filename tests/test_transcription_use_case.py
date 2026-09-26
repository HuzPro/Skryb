from datetime import UTC
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.fakes.dispatcher import DeferredActionDispatcher, InlineActionDispatcher
from voicetotext.core.action_dispatcher import ActionDispatcher
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    ErrorOccurredEvent,
    RecordingCancelledEvent,
    RecordingCapWarningEvent,
    RecordingCappedEvent,
    RecordingPausedEvent,
    RecordingResumedEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    TranscriptionCompletedEvent,
    TranscriptionEmptyEvent,
    TranscriptionFailedEvent,
    TranscriptionStartedEvent,
)
from voicetotext.core.exceptions import AudioRecorderError, EngineSwapError
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.diagnostics.session import SessionDiagnostics
from voicetotext.output.clipboard_write_error import ClipboardWriteError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(data=np.zeros(samples, dtype=np.float32), sample_rate=16000)


def _make_result(
    text: str = "hello world", seconds: float = 0.5,
) -> TranscriptionResult:
    return TranscriptionResult(
        text=text,
        language="en",
        language_probability=0.95,
        transcription_seconds=seconds,
    )


def _build_use_case(
    dispatcher: ActionDispatcher | None = None,
    initial_state: AppState = AppState.IDLE,
    max_recording_seconds: float = 600.0,
    warn_before_cap_seconds: float = 0.0,
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
        diagnostics=MagicMock(spec=SessionDiagnostics),
        max_recording_seconds=lambda: max_recording_seconds,
        warn_before_cap_seconds=lambda: warn_before_cap_seconds,
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

    def test_should_publish_transcription_empty_event_when_result_is_empty(self):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result(text="", seconds=0.27)
        received: list[TranscriptionEmptyEvent] = []
        event_bus.subscribe(TranscriptionEmptyEvent, received.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert len(received) == 1
        assert received[0].transcription_seconds == 0.27

    def test_should_not_publish_empty_event_when_result_has_text(self):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")
        received: list[TranscriptionEmptyEvent] = []
        event_bus.subscribe(TranscriptionEmptyEvent, received.append)

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert received == []

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


class TestTranscriptionDiagnostics:
    def test_should_record_result_after_output_is_delivered(self) -> None:
        use_case, recorder, engine, _, _, _ = _build_use_case()
        audio = _make_audio()
        result = _make_result(
            text="hello", seconds=0.42,
        )
        recorder.stop_recording.return_value = audio
        engine.transcribe.return_value = result

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        diagnostics = use_case._diagnostics
        recorded_audio, recorded_result, requested_at = (
            diagnostics.transcription_finished.call_args.args
        )
        assert recorded_audio is audio
        assert recorded_result is result
        assert requested_at.tzinfo is UTC

    def test_should_record_empty_result(self) -> None:
        use_case, recorder, engine, _, _, _ = _build_use_case()
        audio = _make_audio()
        result = _make_result(text="", seconds=0.31)
        recorder.stop_recording.return_value = audio
        engine.transcribe.return_value = result

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        use_case._diagnostics.transcription_finished.assert_called_once()
        assert use_case._diagnostics.transcription_finished.call_args.args[1] is result


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
        outcome = use_case._diagnostics.transcription_finished.call_args.args[1]
        assert isinstance(outcome, RuntimeError)


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

    def test_should_allow_swap_after_recovering_from_a_failed_load(self):
        # Bug 7: a failed load used to strand the machine in ERROR, where
        # every set_engine raised "Can only swap engine when idle or
        # loading", locking the app until restart. The bootstrap recovery
        # drives ERROR -> IDLE; once idle, the swap must succeed so the
        # user can pick another model without restarting.
        use_case, _, old_engine, _, _, state_machine = _build_use_case()
        state_machine.transition_to(AppState.ERROR)
        state_machine.transition_to(AppState.IDLE)
        new_engine = MagicMock()

        use_case.set_engine(new_engine)  # must not raise

        old_engine.dispose.assert_called_once()

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
    engine via ``set_engine`` and (b) ignore hotkey presses safely -
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
        # it directly - equivalent to the real timer thread firing.
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


class TestRecordingLimitWarning:
    """Half a minute before the limit the pill starts counting down and a
    soft cue plays, so the end of a long dictation is never a surprise."""

    @staticmethod
    def _timers(MockTimer) -> list[float]:
        return [call.args[0] for call in MockTimer.call_args_list]

    @staticmethod
    def _fire_latest_warning(MockTimer) -> None:
        MockTimer.call_args_list[-1].args[1]()

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_the_warning_is_due_the_chosen_time_before_the_limit(self, MockTimer):
        use_case, *_ = _build_use_case(
            max_recording_seconds=600.0, warn_before_cap_seconds=30.0,
        )

        use_case.on_hotkey_activated()

        assert self._timers(MockTimer) == [600.0, 570.0]

    @patch("voicetotext.transcription_use_case.time.monotonic")
    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_the_warning_fires_exactly_once_with_the_time_left(
        self, MockTimer, monotonic,
    ):
        use_case, _, _, _, event_bus, _ = _build_use_case(
            max_recording_seconds=600.0, warn_before_cap_seconds=30.0,
        )
        warnings: list[RecordingCapWarningEvent] = []
        event_bus.subscribe(RecordingCapWarningEvent, warnings.append)
        monotonic.return_value = 1000.0
        use_case.on_hotkey_activated()

        monotonic.return_value = 1570.0
        self._fire_latest_warning(MockTimer)
        self._fire_latest_warning(MockTimer)

        assert [event.remaining_seconds for event in warnings] == [30.0]

    @patch("voicetotext.transcription_use_case.time.monotonic")
    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_paused_time_pushes_the_warning_back(self, MockTimer, monotonic):
        use_case, *_ = _build_use_case(
            max_recording_seconds=600.0, warn_before_cap_seconds=30.0,
        )
        monotonic.return_value = 1000.0
        use_case.on_hotkey_activated()
        monotonic.return_value = 1100.0
        use_case.on_pause_toggled()
        monotonic.return_value = 5000.0

        use_case.on_pause_toggled()

        assert self._timers(MockTimer)[-2:] == [500.0, 470.0]

    @patch("voicetotext.transcription_use_case.time.monotonic")
    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_a_warning_due_while_paused_waits_for_the_resume(self, MockTimer, monotonic):
        use_case, _, _, _, event_bus, _ = _build_use_case(
            max_recording_seconds=600.0, warn_before_cap_seconds=30.0,
        )
        warnings: list[RecordingCapWarningEvent] = []
        event_bus.subscribe(RecordingCapWarningEvent, warnings.append)
        monotonic.return_value = 1000.0
        use_case.on_hotkey_activated()
        monotonic.return_value = 1575.0
        use_case.on_pause_toggled()

        self._fire_latest_warning(MockTimer)
        assert warnings == []
        monotonic.return_value = 9000.0
        use_case.on_pause_toggled()
        self._fire_latest_warning(MockTimer)

        assert self._timers(MockTimer)[-2:] == [25.0, 0.0]
        assert [event.remaining_seconds for event in warnings] == [25.0]

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_a_limit_changed_after_startup_applies_to_the_next_recording(
        self, MockTimer,
    ):
        limits = {"stop": 600.0, "warn": 30.0}
        event_bus = SimpleEventBus()
        recorder = MagicMock()
        recorder.stop_recording.return_value = None
        use_case = RecordAndTranscribeUseCase(
            recorder=recorder,
            engine=MagicMock(),
            output=MagicMock(),
            state_machine=AppStateMachine(event_bus),
            event_bus=event_bus,
            action_dispatcher=InlineActionDispatcher(),
            diagnostics=MagicMock(spec=SessionDiagnostics),
            max_recording_seconds=lambda: limits["stop"],
            warn_before_cap_seconds=lambda: limits["warn"],
        )
        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        limits.update(stop=120.0, warn=10.0)
        use_case.on_hotkey_activated()

        assert self._timers(MockTimer) == [600.0, 570.0, 120.0, 110.0]

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_the_capped_notice_names_the_limit_that_was_in_force(self, MockTimer):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case(
            max_recording_seconds=90.0, warn_before_cap_seconds=30.0,
        )
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("done")
        capped: list[RecordingCappedEvent] = []
        event_bus.subscribe(RecordingCappedEvent, capped.append)
        use_case.on_hotkey_activated()

        MockTimer.call_args_list[0].args[1]()

        assert [event.max_duration_seconds for event in capped] == [90.0]

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_no_warning_is_scheduled_when_it_is_turned_off(self, MockTimer):
        use_case, *_ = _build_use_case(
            max_recording_seconds=600.0, warn_before_cap_seconds=0.0,
        )

        use_case.on_hotkey_activated()

        assert self._timers(MockTimer) == [600.0]

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_a_limit_shorter_than_the_warning_warns_straight_away(self, MockTimer):
        use_case, *_ = _build_use_case(
            max_recording_seconds=20.0, warn_before_cap_seconds=30.0,
        )

        use_case.on_hotkey_activated()

        assert self._timers(MockTimer) == [20.0, 0.0]

    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_stopping_cancels_the_warning_along_with_the_limit(self, MockTimer):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case(
            warn_before_cap_seconds=30.0,
        )
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("done")
        warnings: list[RecordingCapWarningEvent] = []
        event_bus.subscribe(RecordingCapWarningEvent, warnings.append)
        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        self._fire_latest_warning(MockTimer)

        assert MockTimer.return_value.cancel.call_count == 2
        assert warnings == []


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
    the hotkey callback - everything goes through the ActionDispatcher so
    the keyboard hook's callback thread is never blocked by slow work."""

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

        # Transcription is queued but has not run yet - state is TRANSCRIBING.
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

        use_case._diagnostics.dispose.assert_called_once()
        engine.dispose.assert_called_once()


class TestPause:
    def test_pausing_stops_keeping_audio_but_stays_recording(self):
        use_case, recorder, _, _, event_bus, state_machine = _build_use_case()
        paused: list[RecordingPausedEvent] = []
        event_bus.subscribe(RecordingPausedEvent, paused.append)
        use_case.on_hotkey_activated()

        use_case.on_pause_toggled()

        recorder.set_paused.assert_called_once_with(True)
        assert state_machine.state == AppState.RECORDING
        assert len(paused) == 1

    def test_pausing_again_resumes_the_same_recording(self):
        use_case, recorder, _, _, event_bus, _ = _build_use_case()
        resumed: list[RecordingResumedEvent] = []
        event_bus.subscribe(RecordingResumedEvent, resumed.append)
        use_case.on_hotkey_activated()

        use_case.on_pause_toggled()
        use_case.on_pause_toggled()

        assert [call.args for call in recorder.set_paused.call_args_list] == [
            (True,), (False,),
        ]
        assert recorder.start_recording.call_count == 1
        assert len(resumed) == 1

    def test_the_hotkey_finishes_a_paused_recording_and_transcribes_it(self):
        use_case, recorder, engine, output, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("paused then done")
        use_case.on_hotkey_activated()
        use_case.on_pause_toggled()

        use_case.on_hotkey_activated()

        output.output.assert_called_once_with("paused then done")
        assert state_machine.state == AppState.IDLE

    @pytest.mark.parametrize("state", [AppState.IDLE, AppState.LOADING])
    def test_pause_does_nothing_when_not_recording(self, state):
        use_case, recorder, *_ = _build_use_case(initial_state=state)

        use_case.on_pause_toggled()

        recorder.set_paused.assert_not_called()

    @patch("voicetotext.transcription_use_case.time.monotonic")
    @patch("voicetotext.transcription_use_case.threading.Timer")
    def test_paused_time_does_not_count_towards_the_recording_cap(
        self, MockTimer, monotonic,
    ):
        from unittest.mock import ANY

        use_case, *_ = _build_use_case(max_recording_seconds=600.0)
        monotonic.return_value = 1000.0
        use_case.on_hotkey_activated()
        monotonic.return_value = 1100.0
        use_case.on_pause_toggled()
        monotonic.return_value = 5000.0

        use_case.on_pause_toggled()

        assert MockTimer.call_args_list[-1].args == (500.0, ANY)


class TestCancel:
    def test_cancel_discards_the_recording_without_transcribing_or_pasting(self):
        use_case, recorder, engine, output, event_bus, state_machine = (
            _build_use_case()
        )
        cancelled: list[RecordingCancelledEvent] = []
        event_bus.subscribe(RecordingCancelledEvent, cancelled.append)
        recorder.stop_recording.return_value = _make_audio()
        use_case.on_hotkey_activated()

        use_case.on_cancel()

        engine.transcribe.assert_not_called()
        output.output.assert_not_called()
        assert state_machine.state == AppState.IDLE
        assert len(cancelled) == 1

    def test_cancel_keeps_the_audio_in_history(self):
        use_case, recorder, *_ = _build_use_case()
        audio = _make_audio()
        recorder.stop_recording.return_value = audio
        use_case.on_hotkey_activated()

        use_case.on_cancel()

        use_case._diagnostics.recording_cancelled.assert_called_once_with(audio)

    def test_cancel_works_while_paused_and_the_next_recording_starts_unpaused(self):
        use_case, recorder, *_ = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        use_case.on_hotkey_activated()
        use_case.on_pause_toggled()

        use_case.on_cancel()
        use_case.on_hotkey_activated()
        use_case.on_pause_toggled()

        assert recorder.set_paused.call_args_list[-1].args == (True,)

    def test_cancel_does_nothing_while_transcribing(self):
        use_case, recorder, *_ = _build_use_case(
            dispatcher=DeferredActionDispatcher(),
        )
        use_case._state_machine.transition_to(AppState.RECORDING)
        use_case._state_machine.transition_to(AppState.TRANSCRIBING)

        use_case._handle_cancel()

        recorder.stop_recording.assert_not_called()

    def test_cancel_reports_a_recorder_that_fails_to_stop(self):
        use_case, recorder, _, _, event_bus, state_machine = _build_use_case()
        errors: list[ErrorOccurredEvent] = []
        event_bus.subscribe(ErrorOccurredEvent, errors.append)
        recorder.stop_recording.side_effect = AudioRecorderError("device gone")
        use_case.on_hotkey_activated()

        use_case.on_cancel()

        assert state_machine.state == AppState.IDLE
        assert errors and "device gone" in errors[0].error


class TestTranscriptionEstimate:
    def test_announces_the_audio_length_and_the_learnt_estimate(self):
        use_case, recorder, engine, _, event_bus, _ = _build_use_case()
        started: list[TranscriptionStartedEvent] = []
        event_bus.subscribe(TranscriptionStartedEvent, started.append)
        recorder.stop_recording.return_value = _make_audio(samples=32000)
        engine.transcribe.return_value = _make_result()
        use_case._diagnostics.estimate_transcription_seconds.return_value = 0.7

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert started == [
            TranscriptionStartedEvent(
                timestamp=started[0].timestamp,
                audio_seconds=2.0,
                estimated_seconds=0.7,
            ),
        ]


class ThreadPerActionDispatcher:
    def dispatch(self, action) -> None:
        import threading

        threading.Thread(target=action, daemon=True).start()


class TestRetranscribe:
    def _wav(self, tmp_path, seconds: float = 1.0):
        import wave

        path = tmp_path / "0007.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(np.zeros(int(16000 * seconds), dtype="<i2").tobytes())
        return path

    def test_transcribes_a_saved_recording_without_pasting_it(self, tmp_path):
        use_case, _, engine, output, _, _ = _build_use_case()
        engine.transcribe.return_value = _make_result("second try, café ✓")
        results = []

        use_case.retranscribe(self._wav(tmp_path), results.append)

        assert results == ["second try, café ✓"]
        output.output.assert_not_called()
        assert engine.transcribe.call_args.args[0].duration_seconds == pytest.approx(1.0)

    def test_hands_back_the_error_when_the_file_is_gone(self, tmp_path):
        use_case, _, engine, _, _, _ = _build_use_case()
        results = []

        use_case.retranscribe(tmp_path / "missing.wav", results.append)

        assert isinstance(results[0], FileNotFoundError)
        engine.transcribe.assert_not_called()

    def test_hands_back_the_error_when_the_engine_fails(self, tmp_path):
        use_case, _, engine, _, _, state_machine = _build_use_case()
        engine.transcribe.side_effect = RuntimeError("CUDA out of memory")
        results = []

        use_case.retranscribe(self._wav(tmp_path), results.append)

        assert str(results[0]) == "CUDA out of memory"
        assert state_machine.state == AppState.IDLE

    def test_never_runs_the_model_on_two_recordings_at_once(self, tmp_path):
        import threading
        import time

        use_case, recorder, engine, _, _, _ = _build_use_case(
            dispatcher=ThreadPerActionDispatcher(),
        )
        running = 0
        overlapped = []
        lock = threading.Lock()
        finished = threading.Semaphore(0)

        def slow_transcribe(_audio):
            nonlocal running
            with lock:
                running += 1
                overlapped.append(running)
            time.sleep(0.05)
            with lock:
                running -= 1
            return _make_result("done")

        engine.transcribe.side_effect = slow_transcribe
        recorder.stop_recording.return_value = _make_audio()
        wav = self._wav(tmp_path)

        for _ in range(4):
            use_case.retranscribe(wav, lambda _result: finished.release())
        use_case._handle_activate()
        use_case._handle_activate()
        for _ in range(4):
            assert finished.acquire(timeout=5)
        deadline = time.monotonic() + 5
        while engine.transcribe.call_count < 5 and time.monotonic() < deadline:
            time.sleep(0.01)

        assert engine.transcribe.call_count == 5
        assert max(overlapped) == 1


class TestOnePressAlwaysStartsTheNextRecording:
    """The real toggle hotkey wired to the real use case, as bootstrap does."""

    def _wired(self):
        from voicetotext.hotkeys import keys
        from voicetotext.hotkeys.toggle_strategy import ToggleStrategy

        use_case, recorder, engine, _, _, state_machine = _build_use_case()
        recorder.stop_recording.return_value = _make_audio()
        engine.transcribe.return_value = _make_result("hello")
        combo = (keys.HotkeyKey(keys.VK_LCONTROL), keys.from_char("m"))
        strategy = ToggleStrategy(keys=combo)
        strategy.set_callbacks(use_case.on_hotkey_activated, use_case.on_hotkey_deactivated)

        def press() -> None:
            for key in combo:
                strategy.on_key_press(key)
            for key in reversed(combo):
                strategy.on_key_release(key)

        return use_case, state_machine, press

    def test_after_cancelling(self):
        use_case, state_machine, press = self._wired()
        press()
        use_case.on_cancel()

        press()

        assert state_machine.state == AppState.RECORDING

    def test_after_finishing_from_the_pill(self):
        use_case, state_machine, press = self._wired()
        press()
        use_case.on_hotkey_deactivated()

        press()

        assert state_machine.state == AppState.RECORDING

    def test_after_the_length_cap_stopped_it(self):
        use_case, state_machine, press = self._wired()
        press()
        use_case._handle_auto_stop()

        press()

        assert state_machine.state == AppState.RECORDING

    def test_after_a_recording_too_short_to_transcribe(self):
        use_case, state_machine, press = self._wired()
        use_case._recorder.stop_recording.return_value = None
        press()
        press()

        press()

        assert state_machine.state == AppState.RECORDING

    def test_across_many_cancel_and_finish_cycles(self):
        use_case, state_machine, press = self._wired()

        for cycle in range(500):
            press()
            assert state_machine.state == AppState.RECORDING
            if cycle % 3 == 0:
                use_case.on_cancel()
            elif cycle % 3 == 1:
                use_case.on_hotkey_deactivated()
            else:
                press()
            assert state_machine.state == AppState.IDLE
