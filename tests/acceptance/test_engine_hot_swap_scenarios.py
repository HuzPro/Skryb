"""Engine hot-swap - swapping the transcription model at runtime."""

from tests.acceptance.app_fixture import SkrybApp
from tests.builders import a_transcription_result
from tests.fakes import FakeTranscriptionEngine
from skryb.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
)


class TestEngineHotSwap:
    def test_switching_model_while_idle_loads_new_engine(
        self, voice_to_text_app: SkrybApp
    ):
        new_engine = FakeTranscriptionEngine()
        new_engine.will_return(
            a_transcription_result().with_text("from new engine").build()
        )
        voice_to_text_app.next_engine_will_be(new_engine)

        voice_to_text_app.trigger_engine_swap("nemotron-speech-streaming-en-0.6b")

        completed = voice_to_text_app.events_of(EngineLoadCompletedEvent)
        assert len(completed) == 1
        assert completed[0].model_name == "nemotron-speech-streaming-en-0.6b"

    def test_new_engine_is_used_for_next_transcription(
        self, voice_to_text_app: SkrybApp
    ):
        new_engine = FakeTranscriptionEngine()
        new_engine.will_return(
            a_transcription_result().with_text("from new engine").build()
        )
        voice_to_text_app.next_engine_will_be(new_engine)
        voice_to_text_app.trigger_engine_swap("nemotron-speech-streaming-en-0.6b")

        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.typed_text == "from new engine"

    def test_switching_model_while_recording_is_rejected(
        self, voice_to_text_app: SkrybApp
    ):
        voice_to_text_app.press_hotkey()  # state = RECORDING
        new_engine = FakeTranscriptionEngine()
        voice_to_text_app.next_engine_will_be(new_engine)

        voice_to_text_app.trigger_engine_swap("nemotron-speech-streaming-en-0.6b")

        failed = voice_to_text_app.events_of(EngineLoadFailedEvent)
        assert len(failed) == 1
        assert "idle" in failed[0].error.lower()

    def test_old_engine_is_disposed_on_successful_swap(
        self, voice_to_text_app: SkrybApp
    ):
        old_engine = voice_to_text_app.current_engine
        new_engine = FakeTranscriptionEngine()
        new_engine.will_return(a_transcription_result().build())
        voice_to_text_app.next_engine_will_be(new_engine)

        voice_to_text_app.trigger_engine_swap("parakeet-ultra")

        assert old_engine.dispose_called is True

    def test_old_engine_stays_live_when_swap_rejected_mid_recording(
        self, voice_to_text_app: SkrybApp
    ):
        old_engine = voice_to_text_app.current_engine
        old_engine.will_return(
            a_transcription_result().with_text("from old engine").build()
        )
        voice_to_text_app.press_hotkey()  # state = RECORDING
        new_engine = FakeTranscriptionEngine()
        new_engine.will_return(
            a_transcription_result().with_text("from new engine").build()
        )
        voice_to_text_app.next_engine_will_be(new_engine)

        voice_to_text_app.trigger_engine_swap("nemotron-speech-streaming-en-0.6b")

        assert new_engine.dispose_called is True
        assert old_engine.dispose_called is False

        voice_to_text_app.release_hotkey()

        assert voice_to_text_app.typed_text == "from old engine"
