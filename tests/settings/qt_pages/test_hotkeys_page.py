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


class TestHotkeysPageRecordingState:
    def test_should_emit_recording_state_changed_true_when_capture_starts(
        self, page
    ):
        events: list[bool] = []
        page.recording_state_changed.connect(events.append)

        page.hotkey_capture().start_recording()

        assert events == [True]

    def test_should_emit_recording_state_changed_false_when_capture_stops(
        self, page
    ):
        events: list[bool] = []
        page.recording_state_changed.connect(events.append)
        page.hotkey_capture().start_recording()

        page.hotkey_capture().stop_recording()

        assert events == [True, False]


class TestHotkeysPageBridgeForwarding:
    def test_should_pass_capture_bridge_to_capture_widget(self, qtbot):
        from PySide6.QtCore import QObject, Signal

        class FakeBridge(QObject):
            key_pressed = Signal(str)

            def start(self):
                pass

            def stop(self):
                pass

        bridge = FakeBridge()
        page = HotkeysPage(capture_bridge=bridge)
        qtbot.addWidget(page)

        page.hotkey_capture().start_recording()
        bridge.key_pressed.emit("f12")
        page.hotkey_capture().stop_recording()

        assert page.current_combination() == "f12"
