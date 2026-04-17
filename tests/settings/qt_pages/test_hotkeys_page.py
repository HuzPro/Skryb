import pytest

from voicetotext.settings.qt_pages.hotkeys_page import HotkeysPage


@pytest.fixture
def page(qtbot):
    p = HotkeysPage()
    qtbot.addWidget(p)
    return p


class TestHotkeysPageCombination:
    def test_should_start_empty(self, page):
        assert page.current_combination() == ""

    def test_should_reflect_set_combination(self, page):
        page.set_combination("ctrl+alt+m")

        assert page.current_combination() == "ctrl+alt+m"


class TestHotkeysPageMode:
    def test_should_default_to_toggle(self, page):
        assert page.current_mode() == "toggle"

    def test_should_reflect_push_to_talk_when_set(self, page):
        page.set_mode("push_to_talk")

        assert page.current_mode() == "push_to_talk"

    def test_should_round_trip_toggle(self, page):
        page.set_mode("push_to_talk")
        page.set_mode("toggle")

        assert page.current_mode() == "toggle"

    def test_should_ignore_unknown_mode(self, page):
        page.set_mode("push_to_talk")
        page.set_mode("nonsense")

        assert page.current_mode() == "push_to_talk"


class TestHotkeysPageCapture:
    def test_should_update_combination_when_capture_emits(self, page):
        page.hotkey_capture().combination_changed.emit("ctrl+f12")

        assert page.current_combination() == "ctrl+f12"
