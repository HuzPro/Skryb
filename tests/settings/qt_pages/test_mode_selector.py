import pytest

from voicetotext.settings.qt_pages.mode_selector import ModeSelector
from voicetotext.settings.transcription_mode import TranscriptionMode


@pytest.fixture
def selector(qtbot):
    s = ModeSelector()
    qtbot.addWidget(s)
    return s


class TestModeSelector:
    def test_should_default_to_batch(self, selector):
        assert selector.current_mode() is TranscriptionMode.BATCH

    def test_should_reflect_set_mode(self, selector):
        selector.set_mode(TranscriptionMode.STREAMING)

        assert selector.current_mode() is TranscriptionMode.STREAMING

    def test_should_have_one_radio_per_mode(self, selector):
        assert len(selector._buttons) == 2

    def test_should_not_emit_on_programmatic_set(self, qtbot, selector):
        with qtbot.assertNotEmitted(selector.mode_changed):
            selector.set_mode(TranscriptionMode.STREAMING)

    def test_should_emit_mode_changed_on_user_toggle(self, qtbot, selector):
        with qtbot.waitSignal(selector.mode_changed, timeout=500) as signal:
            selector._buttons[TranscriptionMode.STREAMING].setChecked(True)

        assert signal.args == [TranscriptionMode.STREAMING]


class TestModeHint:
    """The hint line explains the active mode and why the model list changes."""

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
        selector._buttons[TranscriptionMode.STREAMING].setChecked(True)

        assert selector.hint_text() != batch_hint


class TestStreamingHardwareGate:
    """On a box with no compatible GPU, streaming cannot be selected."""

    def test_streaming_is_enabled_by_default(self, selector):
        assert selector._buttons[TranscriptionMode.STREAMING].isEnabled() is True

    def test_disable_streaming_disables_only_the_streaming_radio(self, selector):
        selector.disable_streaming("no GPU detected")

        assert selector._buttons[TranscriptionMode.STREAMING].isEnabled() is False
        assert selector._buttons[TranscriptionMode.BATCH].isEnabled() is True

    def test_disable_streaming_explains_why_in_the_tooltip(self, selector):
        selector.disable_streaming("needs a CUDA or ROCm GPU")

        tooltip = selector._buttons[TranscriptionMode.STREAMING].toolTip()
        assert "needs a CUDA or ROCm GPU" in tooltip
