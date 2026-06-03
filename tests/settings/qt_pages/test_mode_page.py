import pytest

from voicetotext.settings.qt_pages.mode_page import ModePage
from voicetotext.settings.transcription_mode import TranscriptionMode


@pytest.fixture
def page(qtbot):
    p = ModePage()
    qtbot.addWidget(p)
    return p


class TestModePage:
    def test_should_default_to_batch(self, page):
        assert page.current_mode() is TranscriptionMode.BATCH

    def test_should_reflect_set_mode(self, page):
        page.set_mode(TranscriptionMode.STREAMING)

        assert page.current_mode() is TranscriptionMode.STREAMING

    def test_should_have_one_radio_per_mode(self, page):
        assert len(page._buttons) == 2

    def test_should_not_emit_on_programmatic_set(self, qtbot, page):
        with qtbot.assertNotEmitted(page.mode_changed):
            page.set_mode(TranscriptionMode.STREAMING)

    def test_should_emit_mode_changed_on_user_toggle(self, qtbot, page):
        with qtbot.waitSignal(page.mode_changed, timeout=500) as signal:
            page._buttons[TranscriptionMode.STREAMING].setChecked(True)

        assert signal.args == [TranscriptionMode.STREAMING]
