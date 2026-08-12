"""Core voice-to-text flow - the user presses a hotkey, speaks, and
their words are typed into the focused window."""

from tests.acceptance.app_fixture import VoiceToTextApp
from voicetotext.core.events import (
    ErrorOccurredEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.state_machine import AppState


class TestVoiceToTextHappyPath:
    def test_user_speaking_types_their_words_into_focused_window(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("hello world")

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.typed_text == "hello world"

    def test_app_returns_to_idle_after_successful_transcription(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.state == AppState.IDLE

    def test_transcription_completed_event_carries_the_typed_text(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("the quick brown fox")

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        completed = voice_to_text_app.events_of(TranscriptionCompletedEvent)
        assert len(completed) == 1
        assert completed[0].text == "the quick brown fox"


class TestSilentRecordingIsIgnored:
    def test_very_short_recording_is_silently_discarded(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.microphone_captures_silence_of(seconds=0.05)

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.typed_text is None
        assert voice_to_text_app.state == AppState.IDLE

    def test_empty_transcription_does_not_type_anything(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("")

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.typed_text is None

    def test_missing_audio_returns_app_to_idle_without_transcribing(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.microphone_captures_nothing()

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.engine_transcribe_calls == 0
        assert voice_to_text_app.state == AppState.IDLE


class TestRecordingFailures:
    def test_microphone_failure_returns_app_to_idle_and_reports_error(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.microphone_fails_to_start_with(
            RuntimeError("mic unavailable")
        )

        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.state == AppState.IDLE
        errors = voice_to_text_app.events_of(ErrorOccurredEvent)
        assert len(errors) == 1
        assert "mic unavailable" in errors[0].error


class TestTranscriptionFailures:
    def test_engine_failure_returns_app_to_idle_and_reports_error(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.engine_fails_with(RuntimeError("cuda oom"))

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.state == AppState.IDLE
        failed = voice_to_text_app.events_of(TranscriptionFailedEvent)
        assert len(failed) == 1
        assert "cuda oom" in failed[0].error

    def test_engine_failure_does_not_type_anything(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.engine_fails_with(RuntimeError("boom"))

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.typed_text is None
