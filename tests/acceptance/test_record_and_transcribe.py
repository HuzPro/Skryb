"""Full-pipeline scenario: press, speak, release → text appears and
the whole event sequence is emitted in order."""

from tests.acceptance.app_fixture import VoiceToTextApp
from voicetotext.core.events import (
    RecordingStartedEvent,
    RecordingStoppedEvent,
    StateChangedEvent,
    TranscriptionCompletedEvent,
)
from voicetotext.core.state_machine import AppState


class TestFullTogglePipeline:
    def test_should_record_transcribe_and_output_text(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("integration test works")

        voice_to_text_app.press_hotkey()
        assert voice_to_text_app.state == AppState.RECORDING

        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.typed_text == "integration test works"
        assert voice_to_text_app.state == AppState.IDLE

        assert len(voice_to_text_app.events_of(RecordingStartedEvent)) == 1
        assert len(voice_to_text_app.events_of(RecordingStoppedEvent)) == 1
        completed = voice_to_text_app.events_of(TranscriptionCompletedEvent)
        assert len(completed) == 1
        assert completed[0].text == "integration test works"

        state_changes = voice_to_text_app.events_of(StateChangedEvent)
        assert [e.new_state for e in state_changes] == [
            "recording",
            "transcribing",
            "idle",
        ]


class TestFullPushToTalkPipeline:
    def test_should_record_on_activate_and_transcribe_on_deactivate(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("push to talk works")

        voice_to_text_app.press_hotkey()
        assert voice_to_text_app.state == AppState.RECORDING

        voice_to_text_app.release_hotkey()

        assert voice_to_text_app.typed_text == "push to talk works"
        assert voice_to_text_app.state == AppState.IDLE
