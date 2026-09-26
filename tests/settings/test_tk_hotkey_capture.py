import pytest

from voicetotext.settings.signal import Signal
from voicetotext.settings.tk_hotkey_capture import TkHotkeyCapture


class FakeCaptureBridge:
    def __init__(self) -> None:
        self.key_pressed = Signal()
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def emit_key(self, name: str) -> None:
        self.key_pressed.emit(name)


@pytest.fixture
def bridge():
    return FakeCaptureBridge()


@pytest.fixture
def widget(tk_root, bridge):
    w = TkHotkeyCapture(tk_root, capture_bridge=bridge)
    yield w
    w.destroy()


class TestTkHotkeyCaptureInitial:
    def test_should_start_empty(self, widget):
        assert widget.combination() == ""

    def test_should_reflect_set_combination(self, widget):
        widget.set_combination("ctrl+alt+m")

        assert widget.combination() == "ctrl+alt+m"

    def test_should_show_record_label_initially(self, widget):
        assert widget.record_button_text() == "Record"


class TestTkHotkeyCaptureRecording:
    def test_should_not_be_recording_initially(self, widget):
        assert widget.is_recording() is False

    def test_should_enter_recording_mode_when_start_recording_called(self, widget):
        widget.start_recording()

        assert widget.is_recording() is True

    def test_should_show_stop_label_while_recording(self, widget):
        widget.start_recording()

        assert widget.record_button_text() == "Stop Recording"

    def test_should_revert_to_record_label_after_stop(self, widget):
        widget.start_recording()
        widget.stop_recording()

        assert widget.record_button_text() == "Record"

    def test_should_record_modifier_then_letter(self, widget, bridge):
        widget.start_recording()

        bridge.emit_key("ctrl")
        bridge.emit_key("m")
        widget.stop_recording()

        assert widget.combination() == "ctrl+m"

    def test_should_record_ctrl_alt_m_as_three_step_sequence(self, widget, bridge):
        widget.start_recording()

        bridge.emit_key("ctrl")
        bridge.emit_key("alt")
        bridge.emit_key("m")
        widget.stop_recording()

        assert widget.combination() == "ctrl+alt+m"

    def test_should_record_arbitrary_length_sequence(self, widget, bridge):
        widget.start_recording()

        for name in ("alt", "1", "2", "3", "4", "5"):
            bridge.emit_key(name)
        widget.stop_recording()

        assert widget.combination() == "alt+1+2+3+4+5"

    def test_should_emit_signal_when_recording_committed(self, widget, bridge):
        received: list[str] = []
        widget.combination_changed.connect(received.append)
        widget.start_recording()

        bridge.emit_key("ctrl")
        bridge.emit_key("alt")
        bridge.emit_key("m")
        widget.stop_recording()

        assert received == ["ctrl+alt+m"]

    def test_should_dedupe_consecutive_identical_keys(self, widget, bridge):
        widget.start_recording()

        bridge.emit_key("ctrl")
        bridge.emit_key("ctrl")
        bridge.emit_key("m")
        widget.stop_recording()

        assert widget.combination() == "ctrl+m"

    def test_should_cancel_recording_on_escape(self, widget, bridge):
        widget.set_combination("ctrl+alt+m")
        widget.start_recording()

        bridge.emit_key("ctrl")
        bridge.emit_key("esc")

        assert widget.is_recording() is False
        assert widget.combination() == "ctrl+alt+m"

    def test_should_not_commit_empty_recording(self, widget):
        widget.set_combination("ctrl+alt+m")
        widget.start_recording()

        widget.stop_recording()

        assert widget.combination() == "ctrl+alt+m"

    def test_should_show_partial_sequence_during_recording(self, widget, bridge):
        widget.start_recording()

        bridge.emit_key("ctrl")
        assert widget.is_recording() is True
        assert widget.display_text() == "Ctrl"

        bridge.emit_key("m")
        assert widget.display_text() == "Ctrl + M"

    @pytest.mark.parametrize(
        ("combo", "shown"),
        [
            ("ctrl+shift+alt+m", "Ctrl + Shift + Alt + M"),
            ("f13", "F13"),
            ("ctrl+mouse4", "Ctrl + Mouse4"),
            ("", ""),
        ],
    )
    def test_a_combination_reads_like_the_key_caps(self, widget, combo, shown):
        widget.set_combination(combo)

        assert widget.display_text() == shown
        assert widget.combination() == combo

    def test_committing_the_shown_text_unchanged_keeps_the_stored_form(self, widget):
        received: list[str] = []
        widget.set_combination("ctrl+shift+alt+m")
        widget.combination_changed.connect(received.append)

        widget.commit_typed_text()
        widget.set_typed_text("  Ctrl + Alt + F12 ")
        widget.commit_typed_text()

        assert received == ["ctrl+alt+f12"]
        assert widget.combination() == "ctrl+alt+f12"
        assert widget.display_text() == "Ctrl + Alt + F12"


class TestTkHotkeyCaptureBridgeLifecycle:
    def test_should_start_bridge_when_recording_starts(self, widget, bridge):
        widget.start_recording()

        assert bridge.start_calls == 1

    def test_should_stop_bridge_when_recording_stops(self, widget, bridge):
        widget.start_recording()
        widget.stop_recording()

        assert bridge.stop_calls == 1

    def test_should_ignore_bridge_keys_when_not_recording(self, widget, bridge):
        bridge.emit_key("ctrl")

        assert widget.combination() == ""


class TestTkHotkeyCaptureRecordingSignals:
    def test_should_emit_recording_started_when_recording_begins(self, widget):
        events: list[bool] = []
        widget.recording_started.connect(lambda: events.append(True))

        widget.start_recording()

        assert events == [True]

    def test_should_emit_recording_stopped_when_recording_ends(self, widget):
        events: list[bool] = []
        widget.recording_stopped.connect(lambda: events.append(True))
        widget.start_recording()

        widget.stop_recording()

        assert events == [True]

    def test_should_emit_recording_stopped_when_recording_cancelled(
        self, widget, bridge
    ):
        events: list[bool] = []
        widget.recording_stopped.connect(lambda: events.append(True))
        widget.start_recording()

        bridge.emit_key("esc")

        assert events == [True]


class TestTkHotkeyCaptureTextInput:
    def test_should_accept_valid_typed_combination(self, widget):
        widget.set_typed_text("ctrl+alt+m")
        widget.commit_typed_text()

        assert widget.combination() == "ctrl+alt+m"

    def test_should_emit_signal_when_typed_combination_committed(self, widget):
        received: list[str] = []
        widget.combination_changed.connect(received.append)

        widget.set_typed_text("alt+f12")
        widget.commit_typed_text()

        assert received == ["alt+f12"]

    def test_should_reject_invalid_typed_combination(self, widget):
        widget.set_combination("ctrl+m")

        widget.set_typed_text("nonsense")
        widget.commit_typed_text()

        assert widget.combination() == "ctrl+m"
        assert widget.has_validation_error() is True

    def test_should_clear_error_when_valid_combination_replaces_invalid(self, widget):
        widget.set_typed_text("nonsense")
        widget.commit_typed_text()
        assert widget.has_validation_error() is True

        widget.set_typed_text("ctrl+m")
        widget.commit_typed_text()

        assert widget.has_validation_error() is False
        assert widget.combination() == "ctrl+m"

    def test_should_not_emit_signal_for_invalid_typed_combination(self, widget):
        received: list[str] = []
        widget.combination_changed.connect(received.append)

        widget.set_typed_text("nonsense")
        widget.commit_typed_text()

        assert received == []

    def test_should_clear_error_when_recording_starts(self, widget):
        widget.set_typed_text("nonsense")
        widget.commit_typed_text()
        assert widget.has_validation_error() is True

        widget.start_recording()

        assert widget.has_validation_error() is False


class TestTkHotkeyCaptureTeardown:
    def test_closing_mid_recording_releases_the_capture_hooks(self, tk_root, bridge):
        stopped: list[bool] = []
        widget = TkHotkeyCapture(tk_root, capture_bridge=bridge)
        widget.recording_stopped.connect(lambda: stopped.append(True))
        widget.start_recording()
        bridge.emit_key("ctrl")

        widget.destroy()

        assert bridge.stop_calls == 1
        assert stopped == [True]

    def test_a_mouse_button_can_be_recorded_with_a_modifier(self, widget, bridge):
        widget.start_recording()
        bridge.emit_key("ctrl")
        bridge.emit_key("mouse4")
        widget.stop_recording()

        assert widget.combination() == "ctrl+mouse4"

    @pytest.mark.parametrize("typed", ["mouse5", "ctrl+f24", "alt+7", "mouse_middle"])
    def test_new_key_names_can_be_typed(self, widget, typed):
        widget.set_typed_text(typed)
        widget.commit_typed_text()

        assert widget.has_validation_error() is False
        assert widget.combination() == typed
