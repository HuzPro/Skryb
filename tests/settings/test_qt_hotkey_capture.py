import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent

from voicetotext.settings.qt_hotkey_capture import QtHotkeyCapture


class FakeCaptureBridge(QObject):
    key_pressed = Signal(str)

    def __init__(self):
        super().__init__()
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1

    def stop(self):
        self.stop_calls += 1

    def emit_key(self, name: str):
        self.key_pressed.emit(name)


@pytest.fixture
def widget(qtbot):
    w = QtHotkeyCapture()
    qtbot.addWidget(w)
    return w


@pytest.fixture
def bridge():
    return FakeCaptureBridge()


@pytest.fixture
def widget_with_bridge(qtbot, bridge):
    w = QtHotkeyCapture(capture_bridge=bridge)
    qtbot.addWidget(w)
    return w


def _send_autorepeat(widget, key):
    event = QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier, "", True, 1)
    QCoreApplication.sendEvent(widget, event)


class TestQtHotkeyCaptureInitial:
    def test_should_start_empty(self, widget):
        assert widget.combination() == ""

    def test_should_reflect_set_combination(self, widget):
        widget.set_combination("ctrl+alt+m")

        assert widget.combination() == "ctrl+alt+m"

    def test_should_show_record_label_initially(self, widget):
        assert widget.record_button_text() == "Record"


class TestQtHotkeyCaptureRecording:
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

    def test_should_record_modifier_as_separate_step(self, qtbot, widget):
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_Control)
        qtbot.keyPress(widget, Qt.Key_M)
        widget.stop_recording()

        assert widget.combination() == "ctrl+m"

    def test_should_record_ctrl_alt_m_as_three_step_sequence(self, qtbot, widget):
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_Control)
        qtbot.keyPress(widget, Qt.Key_Alt)
        qtbot.keyPress(widget, Qt.Key_M)
        widget.stop_recording()

        assert widget.combination() == "ctrl+alt+m"

    def test_should_record_arbitrary_length_sequence(self, qtbot, widget):
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_Alt)
        qtbot.keyPress(widget, Qt.Key_1)
        qtbot.keyPress(widget, Qt.Key_2)
        qtbot.keyPress(widget, Qt.Key_3)
        qtbot.keyPress(widget, Qt.Key_4)
        qtbot.keyPress(widget, Qt.Key_5)
        widget.stop_recording()

        assert widget.combination() == "alt+1+2+3+4+5"

    def test_should_emit_signal_when_recording_committed(self, qtbot, widget):
        received: list[str] = []
        widget.combination_changed.connect(received.append)
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_Control)
        qtbot.keyPress(widget, Qt.Key_Alt)
        qtbot.keyPress(widget, Qt.Key_M)
        widget.stop_recording()

        assert received == ["ctrl+alt+m"]

    def test_should_capture_function_key(self, qtbot, widget):
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_F12)
        widget.stop_recording()

        assert widget.combination() == "f12"

    def test_should_capture_space(self, qtbot, widget):
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_Shift)
        qtbot.keyPress(widget, Qt.Key_Space)
        widget.stop_recording()

        assert widget.combination() == "shift+space"

    def test_should_cancel_recording_on_escape(self, qtbot, widget):
        widget.set_combination("ctrl+alt+m")
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_Control)
        qtbot.keyClick(widget, Qt.Key_Escape)

        assert widget.is_recording() is False
        assert widget.combination() == "ctrl+alt+m"

    def test_should_not_commit_empty_recording(self, qtbot, widget):
        widget.set_combination("ctrl+alt+m")
        widget.start_recording()

        widget.stop_recording()

        assert widget.combination() == "ctrl+alt+m"

    def test_should_show_partial_sequence_during_recording(self, qtbot, widget):
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_Control)

        assert widget.is_recording() is True
        assert widget.display_text() == "ctrl"

        qtbot.keyPress(widget, Qt.Key_M)

        assert widget.display_text() == "ctrl+m"

    def test_should_filter_autorepeat_modifier_events(self, qtbot, widget):
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_Control)
        _send_autorepeat(widget, Qt.Key_Control)
        _send_autorepeat(widget, Qt.Key_Control)
        qtbot.keyPress(widget, Qt.Key_M)
        widget.stop_recording()

        assert widget.combination() == "ctrl+m"

    def test_should_filter_autorepeat_letter_events(self, qtbot, widget):
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_M)
        _send_autorepeat(widget, Qt.Key_M)
        _send_autorepeat(widget, Qt.Key_M)
        widget.stop_recording()

        assert widget.combination() == "m"


class TestQtHotkeyCaptureTextInput:
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


class TestQtHotkeyCaptureRecordingSignals:
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
        self, qtbot, widget
    ):
        events: list[bool] = []
        widget.recording_stopped.connect(lambda: events.append(True))
        widget.start_recording()

        qtbot.keyClick(widget, Qt.Key_Escape)

        assert events == [True]


class TestQtHotkeyCaptureBridge:
    def test_should_start_bridge_when_recording_starts(
        self, widget_with_bridge, bridge
    ):
        widget_with_bridge.start_recording()

        assert bridge.start_calls == 1

    def test_should_stop_bridge_when_recording_stops(
        self, widget_with_bridge, bridge
    ):
        widget_with_bridge.start_recording()
        widget_with_bridge.stop_recording()

        assert bridge.stop_calls == 1

    def test_should_record_key_emitted_by_bridge(
        self, widget_with_bridge, bridge
    ):
        widget_with_bridge.start_recording()

        bridge.emit_key("ctrl")
        bridge.emit_key("alt")
        bridge.emit_key("m")
        widget_with_bridge.stop_recording()

        assert widget_with_bridge.combination() == "ctrl+alt+m"

    def test_should_cancel_recording_when_bridge_emits_esc(
        self, widget_with_bridge, bridge
    ):
        widget_with_bridge.set_combination("ctrl+m")
        widget_with_bridge.start_recording()

        bridge.emit_key("ctrl")
        bridge.emit_key("esc")

        assert widget_with_bridge.is_recording() is False
        assert widget_with_bridge.combination() == "ctrl+m"

    def test_should_ignore_bridge_keys_when_not_recording(
        self, widget_with_bridge, bridge
    ):
        bridge.emit_key("ctrl")

        assert widget_with_bridge.combination() == ""
