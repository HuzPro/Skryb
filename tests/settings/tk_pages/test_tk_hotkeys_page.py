import pytest

from voicetotext.settings.signal import Signal
from voicetotext.settings.tk_pages.hotkeys_page import TkHotkeysPage


class FakeCaptureBridge:
    def __init__(self) -> None:
        self.key_pressed = Signal()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def emit_key(self, name: str) -> None:
        self.key_pressed.emit(name)


@pytest.fixture
def bridge():
    return FakeCaptureBridge()


@pytest.fixture
def page(tk_root, bridge):
    widget = TkHotkeysPage(tk_root, capture_bridge=bridge)
    yield widget
    widget.destroy()


class TestTkHotkeysPageCombination:
    def test_should_start_empty(self, page):
        assert page.current_combination() == ""

    def test_should_reflect_set_combination(self, page):
        page.set_combination("ctrl+alt+m")

        assert page.current_combination() == "ctrl+alt+m"


class TestTkHotkeysPageMode:
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


class TestTkHotkeysPageCapture:
    def test_should_update_combination_when_capture_emits(self, page):
        page.hotkey_capture().combination_changed.emit("ctrl+f12")

        assert page.current_combination() == "ctrl+f12"


class TestTkHotkeysPageRecordingState:
    def test_should_emit_recording_state_true_when_capture_starts(self, page):
        events: list[bool] = []
        page.recording_state_changed.connect(events.append)

        page.hotkey_capture().start_recording()

        assert events == [True]

    def test_should_emit_recording_state_false_when_capture_stops(self, page):
        events: list[bool] = []
        page.recording_state_changed.connect(events.append)
        page.hotkey_capture().start_recording()

        page.hotkey_capture().stop_recording()

        assert events == [True, False]


class TestTkHotkeysPageBridgeForwarding:
    def test_should_drive_the_injected_capture_bridge(self, tk_root, bridge):
        page = TkHotkeysPage(tk_root, capture_bridge=bridge)

        page.hotkey_capture().start_recording()
        bridge.emit_key("f12")
        page.hotkey_capture().stop_recording()

        assert page.current_combination() == "f12"

        page.destroy()
