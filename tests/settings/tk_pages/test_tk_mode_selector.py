import pytest

from voicetotext.settings.tk_pages.mode_selector import TkModeSelector
from voicetotext.settings.transcription_mode import TranscriptionMode


@pytest.fixture
def selector(tk_root):
    widget = TkModeSelector(tk_root)
    yield widget
    widget.destroy()


class TestTkModeSelector:
    def test_should_default_to_batch(self, selector):
        assert selector.current_mode() is TranscriptionMode.BATCH

    def test_should_reflect_set_mode(self, selector):
        selector.set_mode(TranscriptionMode.STREAMING)

        assert selector.current_mode() is TranscriptionMode.STREAMING

    def test_should_have_one_radio_per_mode(self, selector):
        modes = [selector.radio_button(m) for m in TranscriptionMode]

        assert len(modes) == 2

    def test_should_not_emit_on_programmatic_set(self, selector):
        received: list[TranscriptionMode] = []
        selector.mode_changed.connect(received.append)

        selector.set_mode(TranscriptionMode.STREAMING)

        assert received == []

    def test_should_emit_mode_changed_on_user_toggle(self, selector):
        received: list[TranscriptionMode] = []
        selector.mode_changed.connect(received.append)

        selector.radio_button(TranscriptionMode.STREAMING).invoke()

        assert received == [TranscriptionMode.STREAMING]


class TestTkModeHint:
    def test_batch_hint_describes_record_then_transcribe(self, selector):
        hint = selector.hint_text().lower()

        assert "record" in hint
        assert "transcribe" in hint

    def test_streaming_hint_mentions_live_typing_and_gpu(self, selector):
        selector.set_mode(TranscriptionMode.STREAMING)
        hint = selector.hint_text().lower()

        assert "live" in hint
        assert "gpu" in hint

    def test_streaming_hint_explains_the_filtered_model_list(self, selector):
        selector.set_mode(TranscriptionMode.STREAMING)

        assert "streaming models" in selector.hint_text().lower()

    def test_hint_follows_user_toggles(self, selector):
        batch_hint = selector.hint_text()

        selector.radio_button(TranscriptionMode.STREAMING).invoke()

        assert selector.hint_text() != batch_hint


class TestTkStreamingHardwareGate:
    def test_streaming_is_enabled_by_default(self, selector):
        assert selector.is_streaming_enabled() is True

    def test_disable_streaming_disables_only_the_streaming_radio(self, selector):
        selector.disable_streaming("no GPU detected")

        assert selector.is_streaming_enabled() is False
        assert "disabled" not in selector.radio_button(
            TranscriptionMode.BATCH
        ).state()

    def test_disable_streaming_records_the_reason(self, selector):
        selector.disable_streaming("needs a CUDA or ROCm GPU")

        assert "needs a CUDA or ROCm GPU" in selector.streaming_reason()
