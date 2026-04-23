import pytest
from PySide6.QtCore import Qt

from voicetotext.settings.qt_hotkey_capture import QtHotkeyCapture


@pytest.fixture
def widget(qtbot):
    w = QtHotkeyCapture()
    qtbot.addWidget(w)
    return w


class TestQtHotkeyCaptureInitial:
    def test_should_start_empty(self, widget):
        assert widget.combination() == ""

    def test_should_reflect_set_combination(self, widget):
        widget.set_combination("ctrl+alt+m")

        assert widget.combination() == "ctrl+alt+m"


class TestQtHotkeyCaptureRecording:
    def test_should_not_be_recording_initially(self, widget):
        assert widget.is_recording() is False

    def test_should_enter_recording_mode_when_start_recording_called(self, widget):
        widget.start_recording()

        assert widget.is_recording() is True

    def test_should_capture_ctrl_alt_m_combination(self, qtbot, widget):
        widget.start_recording()
        qtbot.keyClick(
            widget, Qt.Key_M, Qt.ControlModifier | Qt.AltModifier
        )

        assert widget.combination() == "ctrl+alt+m"

    def test_should_exit_recording_mode_after_combo_captured(self, qtbot, widget):
        widget.start_recording()
        qtbot.keyClick(widget, Qt.Key_M, Qt.ControlModifier)

        assert widget.is_recording() is False

    def test_should_emit_signal_when_combination_captured(self, qtbot, widget):
        received: list[str] = []
        widget.combination_changed.connect(received.append)
        widget.start_recording()

        qtbot.keyClick(widget, Qt.Key_M, Qt.ControlModifier | Qt.AltModifier)

        assert received == ["ctrl+alt+m"]

    def test_should_capture_function_key(self, qtbot, widget):
        widget.start_recording()
        qtbot.keyClick(widget, Qt.Key_F12)

        assert widget.combination() == "f12"

    def test_should_capture_space_with_modifier(self, qtbot, widget):
        widget.start_recording()
        qtbot.keyClick(widget, Qt.Key_Space, Qt.ShiftModifier)

        assert widget.combination() == "shift+space"

    def test_should_cancel_recording_on_escape(self, qtbot, widget):
        widget.set_combination("ctrl+alt+m")
        widget.start_recording()

        qtbot.keyClick(widget, Qt.Key_Escape)

        assert widget.is_recording() is False
        assert widget.combination() == "ctrl+alt+m"

    def test_should_stay_recording_when_only_modifier_pressed(self, qtbot, widget):
        widget.start_recording()

        qtbot.keyPress(widget, Qt.Key_Control)

        assert widget.is_recording() is True
        assert widget.combination() == ""
