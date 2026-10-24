"""Paste-last, copy-last and transcribing a cancelled or failed recording again."""

from unittest.mock import MagicMock

import numpy as np

from tests.fakes import FakeTextOutputHandler, FakeTranscriptionEngine
from tests.fakes.dispatcher import InlineActionDispatcher
from skryb.core.event_bus import SimpleEventBus
from skryb.core.events import (
    NoLastDictationEvent,
    RecordingCancelledEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from skryb.core.state_machine import AppState, AppStateMachine
from skryb.diagnostics.session import SessionDiagnostics
from skryb.output.last_dictation import LastDictation, LastDictationDecorator
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.result import TranscriptionResult
from skryb.transcription_use_case import RecordAndTranscribeUseCase
from skryb.tray.app import SystemTrayApp


def _audio(seconds: float) -> AudioSegment:
    samples = int(16000 * seconds)
    return AudioSegment(
        data=np.linspace(-0.2, 0.2, samples, dtype=np.float32), sample_rate=16000,
    )


def _result(text: str) -> TranscriptionResult:
    return TranscriptionResult(
        text=text, language="en", language_probability=0.97, transcription_seconds=0.3,
    )


class App:
    """The use case with its output chain split the way bootstrap splits it."""

    def __init__(self) -> None:
        self.event_bus = SimpleEventBus()
        self.state_machine = AppStateMachine(self.event_bus, initial_state=AppState.IDLE)
        self.recorder = MagicMock()
        self.engine = FakeTranscriptionEngine()
        self.pasted = FakeTextOutputHandler()
        self.copied = FakeTextOutputHandler()
        self.diagnostics = MagicMock(spec=SessionDiagnostics)
        self.diagnostics.estimate_transcription_seconds.return_value = None
        last = LastDictation()
        last.route(paste=self.pasted, copy=self.copied)
        self.use_case = RecordAndTranscribeUseCase(
            recorder=self.recorder,
            engine=self.engine,
            output=LastDictationDecorator(self.pasted, last),
            state_machine=self.state_machine,
            event_bus=self.event_bus,
            action_dispatcher=InlineActionDispatcher(),
            diagnostics=self.diagnostics,
            last_dictation=last,
        )
        self.events: list[object] = []
        for event_type in (
            NoLastDictationEvent, RecordingCancelledEvent,
            TranscriptionCompletedEvent, TranscriptionFailedEvent,
        ):
            self.event_bus.subscribe(event_type, self.events.append)

    def dictate(self, audio: AudioSegment, text: str | Exception) -> None:
        if isinstance(text, Exception):
            self.engine.will_fail(text)
        else:
            self.engine.will_return(_result(text))
        self.recorder.stop_recording.return_value = audio
        self.use_case.on_hotkey_activated()
        self.use_case.on_hotkey_activated()

    def cancel(self, audio: AudioSegment | None) -> None:
        self.recorder.stop_recording.return_value = audio
        self.use_case.on_hotkey_activated()
        self.use_case.on_cancel()

    def of_type(self, event_type: type) -> list:
        return [event for event in self.events if isinstance(event, event_type)]


def test_paste_last_resends_the_final_text_and_copy_last_only_writes_the_clipboard() -> None:
    app = App()
    app.dictate(_audio(2.0), "Ship the build — naïve café ✓")

    app.use_case.paste_last()
    app.use_case.copy_last()

    assert app.pasted.outputs == ["Ship the build — naïve café ✓"] * 2
    assert app.copied.outputs == ["Ship the build — naïve café ✓"]
    assert app.of_type(NoLastDictationEvent) == []


def test_copy_last_leaves_out_the_spaces_that_fitted_it_to_the_caret_and_paste_last_keeps_them(
) -> None:
    app = App()
    app.dictate(_audio(2.0), " and then naïve café ✓ 東京. ")

    app.use_case.paste_last()
    app.use_case.copy_last()

    assert app.pasted.outputs == [" and then naïve café ✓ 東京. "] * 2
    assert app.copied.outputs == ["and then naïve café ✓ 東京."]


def test_asking_for_the_last_dictation_before_any_notifies_once_per_ask() -> None:
    app = App()

    app.use_case.paste_last()
    app.use_case.copy_last()

    assert app.pasted.outputs == [] and app.copied.outputs == []
    assert len(app.of_type(NoLastDictationEvent)) == 2


def test_an_empty_result_does_not_replace_the_last_dictation() -> None:
    app = App()
    app.dictate(_audio(1.5), "Keep this one.")
    app.dictate(_audio(1.5), "   ")

    app.use_case.paste_last()

    assert app.pasted.outputs == ["Keep this one.", "Keep this one."]


def test_a_cancelled_recording_can_still_be_transcribed_and_pasted() -> None:
    app = App()
    kept = _audio(3.0)
    app.cancel(kept)
    app.engine.will_return(_result("I did mean to send this"))

    app.use_case.retry_last_recording()

    assert app.of_type(RecordingCancelledEvent)[0].recoverable is True
    assert app.engine.transcribe_calls == [kept]
    assert app.pasted.outputs == ["I did mean to send this"]
    assert app.of_type(TranscriptionCompletedEvent)[-1].text == "I did mean to send this"
    assert app.state_machine.state == AppState.IDLE


def test_transcribing_it_again_a_second_time_does_nothing() -> None:
    app = App()
    app.cancel(_audio(3.0))
    app.engine.will_return(_result("once"))

    app.use_case.retry_last_recording()
    app.use_case.retry_last_recording()

    assert app.pasted.outputs == ["once"]


def test_a_cancel_with_no_usable_audio_offers_nothing() -> None:
    for audio in (None, _audio(0.05)):
        app = App()
        app.cancel(audio)
        app.use_case.retry_last_recording()

        assert app.of_type(RecordingCancelledEvent)[0].recoverable is False
        assert app.engine.transcribe_calls == []


def test_a_failed_transcription_can_be_retried_and_a_second_failure_is_offered_again(
) -> None:
    app = App()
    audio = _audio(4.0)
    app.dictate(audio, RuntimeError("CUDA failed with error out of memory"))

    app.use_case.retry_last_recording()
    app.engine.will_return(_result("worked the third time"))
    app.use_case.retry_last_recording()

    failures = app.of_type(TranscriptionFailedEvent)
    assert [failure.recoverable for failure in failures] == [True, True]
    assert app.engine.transcribe_calls == [audio, audio, audio]
    assert app.pasted.outputs == ["worked the third time"]
    assert app.state_machine.state == AppState.IDLE


def test_the_tray_icon_leaves_the_error_colour_after_a_successful_retry() -> None:
    app = App()
    icon = MagicMock()
    icon.run.side_effect = lambda setup: setup(icon)
    SystemTrayApp(
        use_case=app.use_case,
        state_machine=app.state_machine,
        event_bus=app.event_bus,
        hotkey_listener=MagicMock(),
        tray_factory=lambda **_kwargs: icon,
    ).run()
    app.dictate(_audio(4.0), RuntimeError("CUDA failed with error out of memory"))
    app.use_case.retry_last_recording()
    assert icon.update_state.call_args.args == (AppState.ERROR, "Error")

    app.engine.will_return(_result("worked the third time"))
    app.use_case.retry_last_recording()

    assert app.pasted.outputs == ["worked the third time"]
    assert icon.update_state.call_args.args == (AppState.IDLE, "Idle - ready")


def test_a_retry_failure_never_disturbs_a_recording_in_progress() -> None:
    app = App()
    app.dictate(_audio(2.0), RuntimeError("model is still loading"))
    app.use_case.on_hotkey_activated()
    assert app.state_machine.state == AppState.RECORDING

    app.use_case.retry_last_recording()

    assert app.state_machine.state == AppState.RECORDING
    assert len(app.of_type(TranscriptionFailedEvent)) == 2
