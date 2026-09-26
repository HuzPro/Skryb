from unittest.mock import MagicMock

import numpy as np
import pytest

from tests.fakes import FakeTextOutputHandler, InlineActionDispatcher
from tests.fakes.writing import FakeClock, ScriptedClient
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import RecordingStartedEvent, WritingNoticeEvent
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.diagnostics.session import SessionDiagnostics
from voicetotext.edit_capture.window import ForegroundWindow
from voicetotext.output.caret_context import NO_CONTEXT, CaretContext
from voicetotext.output.dictionary_config import DictionaryConfig
from voicetotext.output.text_filter_decorator import TextFilterDecorator
from voicetotext.output.text_pipeline import TextPipeline
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase
from voicetotext.writing.client import WritingUnavailable
from voicetotext.writing.voice_edit import (
    MAX_SELECTION_CHARS,
    Selection,
    VoiceEditError,
    VoiceEditor,
)

_SELECTED = "Please send the draft to Ayesha on Thursday at 3 PM. Keep the café budget."
_WORD = ForegroundWindow(hwnd=21, pid=8, process="winword.exe", title="Draft - Word")
_OTHER = ForegroundWindow(hwnd=22, pid=9, process="slack.exe", title="Slack")


class Desktop:
    """What the editor can read: the foreground window and the caret."""

    def __init__(self, selected: str = _SELECTED, window: ForegroundWindow = _WORD) -> None:
        self.window = window
        self.context = CaretContext(before="Hi. ", selected=selected, after="")

    def read_context(self) -> CaretContext:
        return self.context

    def read_window(self) -> ForegroundWindow:
        return self.window


def _editor(desktop: Desktop, client: ScriptedClient, deadline: float = 4.0) -> VoiceEditor:
    return VoiceEditor(
        client,
        deadline=lambda: deadline,
        read_context=desktop.read_context,
        read_window=desktop.read_window,
        clock=client.clock or FakeClock(),
    )


class TestSelection:
    @pytest.mark.parametrize("selected", ["", "   \n "], ids=["nothing", "whitespace"])
    def test_no_selection_means_nothing_to_edit(self, selected):
        assert _editor(Desktop(selected), ScriptedClient()).capture() is None

    def test_a_password_manager_selection_is_never_edited(self):
        desktop = Desktop(window=ForegroundWindow(hwnd=3, pid=4, process="1password.exe", title="1Password"))

        assert _editor(desktop, ScriptedClient()).capture() is None

    def test_an_oversized_selection_is_refused(self):
        editor = _editor(Desktop("é" * (MAX_SELECTION_CHARS + 1)), ScriptedClient())

        with pytest.raises(VoiceEditError, match="Selection too long to edit"):
            editor.capture()

    def test_a_selection_at_the_limit_is_accepted(self):
        text = "Ünïcode ✓ " * (MAX_SELECTION_CHARS // 10)

        assert _editor(Desktop(text), ScriptedClient()).capture() == Selection(text, _WORD)


class TestReplacement:
    def test_the_model_rewrites_the_selection_as_instructed(self):
        desktop = Desktop()
        client = ScriptedClient(answer="Please send the draft to Ayesha on Friday at 3 PM.")
        editor = _editor(desktop, client)
        selection = editor.capture()
        assert selection is not None

        replacement = editor.replacement(selection, " change Thursday to Friday ")

        assert replacement == "Please send the draft to Ayesha on Friday at 3 PM."
        assert client.requests[0][1] == {
            "selected_text": _SELECTED, "instruction": "change Thursday to Friday",
        }

    def test_a_changed_foreground_window_aborts_the_replacement(self):
        desktop = Desktop()
        editor = _editor(desktop, ScriptedClient(answer="Shorter."))
        selection = editor.capture()
        assert selection is not None
        desktop.window = _OTHER

        with pytest.raises(VoiceEditError, match="window changed"):
            editor.replacement(selection, "make it shorter")

    def test_a_changed_selection_aborts_the_replacement(self):
        desktop = Desktop()
        editor = _editor(desktop, ScriptedClient(answer="Shorter."))
        selection = editor.capture()
        assert selection is not None
        desktop.context = CaretContext(before="", selected="something else", after="")

        with pytest.raises(VoiceEditError, match="selection changed"):
            editor.replacement(selection, "make it shorter")

    def test_an_unreadable_selection_still_edits_in_the_same_window_with_the_same_title(self):
        desktop = Desktop()
        editor = _editor(desktop, ScriptedClient(answer="Shorter."))
        selection = editor.capture()
        assert selection is not None
        desktop.context = NO_CONTEXT

        assert editor.replacement(selection, "make it shorter") == "Shorter."

        desktop.window = ForegroundWindow(hwnd=21, pid=8, process="winword.exe", title="Other doc - Word")
        with pytest.raises(VoiceEditError, match="window changed"):
            editor.replacement(selection, "make it shorter")

    def test_a_model_that_is_not_running_leaves_the_selection_alone(self):
        editor = _editor(Desktop(), ScriptedClient(error=WritingUnavailable("the writing model is not running")))
        selection = editor.capture()
        assert selection is not None

        with pytest.raises(VoiceEditError, match="left unchanged"):
            editor.replacement(selection, "make it shorter")

    def test_a_late_answer_is_discarded(self):
        clock = FakeClock()
        editor = _editor(Desktop(), ScriptedClient(answer="Late.", clock=clock, takes=5.0), deadline=4.0)
        selection = editor.capture()
        assert selection is not None

        with pytest.raises(VoiceEditError, match="longer than 4 s"):
            editor.replacement(selection, "make it shorter")


class TestVoiceEditInTheUseCase:
    def _use_case(self, desktop: Desktop, client: ScriptedClient):
        event_bus = SimpleEventBus()
        state_machine = AppStateMachine(event_bus, initial_state=AppState.IDLE)
        recorder = MagicMock()
        recorder.stop_recording.return_value = AudioSegment(
            data=np.zeros(16000, dtype=np.float32), sample_rate=16000,
        )
        engine = MagicMock()
        engine.transcribe.return_value = TranscriptionResult(
            text="change Thursday to Friday", language="en",
            language_probability=0.9, transcription_seconds=0.2,
        )
        output = FakeTextOutputHandler()
        shouting = TextFilterDecorator(output, TextPipeline((str.upper,)))
        diagnostics = MagicMock(spec=SessionDiagnostics)
        diagnostics.estimate_transcription_seconds.return_value = 0.5
        use_case = RecordAndTranscribeUseCase(
            recorder=recorder,
            engine=engine,
            output=shouting,
            state_machine=state_machine,
            event_bus=event_bus,
            action_dispatcher=InlineActionDispatcher(),
            diagnostics=diagnostics,
            voice_edit=_editor(desktop, client),
            writing_seconds=lambda audio_seconds: 0.25 * audio_seconds,
        )
        notices: list[str] = []
        captions: list[str] = []
        event_bus.subscribe(WritingNoticeEvent, lambda event: notices.append(event.message))
        event_bus.subscribe(RecordingStartedEvent, lambda event: captions.append(event.caption))
        return use_case, output, diagnostics, notices, captions, state_machine

    def test_no_selection_starts_an_ordinary_dictation(self):
        use_case, output, diagnostics, notices, captions, _ = self._use_case(
            Desktop(selected=""), ScriptedClient(),
        )

        use_case.on_edit_hotkey_activated()
        use_case.on_edit_hotkey_activated()

        assert captions == [""]
        assert output.outputs == ["CHANGE THURSDAY TO FRIDAY"]
        assert diagnostics.transcription_finished.call_args.kwargs["kind"] == "dictation"
        assert notices == []

    def test_an_edit_pastes_the_replacement_past_the_pipeline_and_is_logged_as_an_edit(self):
        use_case, output, diagnostics, notices, captions, state_machine = self._use_case(
            Desktop(), ScriptedClient(answer="Please send the draft to Ayesha on Friday at 3 PM."),
        )

        use_case.on_edit_hotkey_activated()
        use_case.on_edit_hotkey_deactivated()

        assert captions == ["Editing selection"]
        assert output.outputs == ["Please send the draft to Ayesha on Friday at 3 PM."]
        assert diagnostics.transcription_finished.call_args.kwargs["kind"] == "edit"
        assert notices == []
        assert state_machine.state is AppState.IDLE

    def test_a_changed_foreground_window_aborts_the_replacement_with_one_notice(self):
        desktop = Desktop()
        client = ScriptedClient(answer="Shorter.")
        use_case, output, _, notices, _, _ = self._use_case(desktop, client)

        use_case.on_edit_hotkey_activated()
        desktop.window = _OTHER
        use_case.on_edit_hotkey_activated()

        assert output.outputs == []
        assert len(notices) == 1 and "window changed" in notices[0]

    def test_an_oversized_selection_is_refused_without_recording(self):
        use_case, output, _, notices, captions, state_machine = self._use_case(
            Desktop("x" * (MAX_SELECTION_CHARS + 1)), ScriptedClient(),
        )

        use_case.on_edit_hotkey_activated()

        assert state_machine.state is AppState.IDLE
        assert captions == []
        assert notices == ["Selection too long to edit"]
        assert output.outputs == []

    def test_cancelling_an_edit_forgets_the_selection(self):
        use_case, output, _, _, captions, _ = self._use_case(Desktop(), ScriptedClient(answer="Nope."))

        use_case.on_edit_hotkey_activated()
        use_case.on_cancel()
        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert captions == ["Editing selection", ""]
        assert output.outputs == ["CHANGE THURSDAY TO FRIDAY"]

    def test_the_pill_estimate_includes_the_writing_time(self):
        from voicetotext.core.events import TranscriptionStartedEvent

        use_case, _, _, _, _, _ = self._use_case(Desktop(selected=""), ScriptedClient())
        estimates: list[float | None] = []
        use_case._event_bus.subscribe(
            TranscriptionStartedEvent, lambda event: estimates.append(event.estimated_seconds),
        )

        use_case.on_hotkey_activated()
        use_case.on_hotkey_activated()

        assert estimates == [pytest.approx(0.5 + 0.25)]
