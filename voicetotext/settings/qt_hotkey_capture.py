from typing import Protocol

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from voicetotext.hotkeys.key_parser import parse_hotkey_combo


class CaptureBridge(Protocol):
    key_pressed: Signal

    def start(self) -> None: ...
    def stop(self) -> None: ...


_MODIFIER_KEYS: dict[int, str] = {
    Qt.Key_Control: "ctrl",
    Qt.Key_Alt: "alt",
    Qt.Key_AltGr: "alt",
    Qt.Key_Shift: "shift",
    Qt.Key_Meta: "cmd",
    Qt.Key_Super_L: "cmd",
    Qt.Key_Super_R: "cmd",
}

_NAMED_KEYS: dict[int, str] = {
    Qt.Key_Space: "space",
    Qt.Key_F1: "f1",
    Qt.Key_F2: "f2",
    Qt.Key_F3: "f3",
    Qt.Key_F4: "f4",
    Qt.Key_F5: "f5",
    Qt.Key_F6: "f6",
    Qt.Key_F7: "f7",
    Qt.Key_F8: "f8",
    Qt.Key_F9: "f9",
    Qt.Key_F10: "f10",
    Qt.Key_F11: "f11",
    Qt.Key_F12: "f12",
}

_KEY_A = 0x41
_KEY_Z = 0x5A
_KEY_0 = 0x30
_KEY_9 = 0x39

_RECORD_LABEL = "Record"
_STOP_LABEL = "Stop Recording"
_DEFAULT_PLACEHOLDER = "Type a combination, or click Record"
_RECORDING_PLACEHOLDER = "Press keys, then click Stop"


class QtHotkeyCapture(QWidget):

    combination_changed = Signal(str)
    recording_started = Signal()
    recording_stopped = Signal()

    def __init__(
        self,
        parent: QWidget | None = None,
        capture_bridge: CaptureBridge | None = None,
    ) -> None:
        super().__init__(parent)
        self._combination: str = ""
        self._recording: bool = False
        self._recorded_parts: list[str] = []
        self._validation_error: str = ""
        self._capture_bridge = capture_bridge

        self._line_edit = QLineEdit()
        self._line_edit.setPlaceholderText(_DEFAULT_PLACEHOLDER)
        self._line_edit.editingFinished.connect(self._on_text_editing_finished)

        self._record_button = QPushButton(_RECORD_LABEL)
        self._record_button.setCheckable(True)
        self._record_button.toggled.connect(self._on_record_toggled)

        self._error_label = QLabel("")
        self._error_label.setObjectName("hotkey-error")
        self._error_label.setVisible(False)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._line_edit, stretch=1)
        row.addWidget(self._record_button)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(row)
        outer.addWidget(self._error_label)

        self.setFocusPolicy(Qt.StrongFocus)

        if self._capture_bridge is not None:
            self._capture_bridge.key_pressed.connect(self._on_bridge_key)

    def combination(self) -> str:
        return self._combination

    def set_combination(self, combo: str) -> None:
        self._combination = combo
        self._line_edit.setText(combo)
        self._clear_error()

    def is_recording(self) -> bool:
        return self._recording

    def display_text(self) -> str:
        return self._line_edit.text()

    def record_button_text(self) -> str:
        return self._record_button.text()

    def has_validation_error(self) -> bool:
        return bool(self._validation_error)

    def validation_error(self) -> str:
        return self._validation_error

    def start_recording(self) -> None:
        if not self._record_button.isChecked():
            self._record_button.setChecked(True)

    def stop_recording(self) -> None:
        if self._record_button.isChecked():
            self._record_button.setChecked(False)

    def set_typed_text(self, text: str) -> None:
        self._line_edit.setText(text)

    def commit_typed_text(self) -> None:
        self._on_text_editing_finished()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._recording:
            super().keyPressEvent(event)
            return
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key_Escape:
            self._cancel_recording()
            return
        name = self._key_name_for_raw(event)
        if name is None:
            return
        self._recorded_parts.append(name)
        self._line_edit.setText("+".join(self._recorded_parts))

    def _on_record_toggled(self, checked: bool) -> None:
        self._recording = checked
        if checked:
            self._begin_recording()
        else:
            self._end_recording()

    def _begin_recording(self) -> None:
        self._recorded_parts = []
        self._clear_error()
        self._line_edit.setReadOnly(True)
        self._line_edit.setText("")
        self._line_edit.setPlaceholderText(_RECORDING_PLACEHOLDER)
        self._record_button.setText(_STOP_LABEL)
        if self._capture_bridge is not None:
            self._capture_bridge.start()
        else:
            self.setFocus()
            self.grabKeyboard()
        self.recording_started.emit()

    def _end_recording(self) -> None:
        if self._capture_bridge is not None:
            self._capture_bridge.stop()
        else:
            self.releaseKeyboard()
        self._line_edit.setReadOnly(False)
        self._line_edit.setPlaceholderText(_DEFAULT_PLACEHOLDER)
        self._record_button.setText(_RECORD_LABEL)
        if self._recorded_parts:
            combo = "+".join(self._recorded_parts)
            self._set_combination_internal(combo)
        else:
            self._line_edit.setText(self._combination)
        self.recording_stopped.emit()

    def _cancel_recording(self) -> None:
        self._recorded_parts = []
        self._record_button.setChecked(False)

    def _on_bridge_key(self, name: str) -> None:
        if not self._recording:
            return
        if name == "esc":
            self._cancel_recording()
            return
        self._recorded_parts.append(name)
        self._line_edit.setText("+".join(self._recorded_parts))

    def _on_text_editing_finished(self) -> None:
        if self._recording:
            return
        text = self._line_edit.text().strip()
        if text == self._combination:
            self._clear_error()
            return
        if not text:
            self._set_combination_internal("")
            self._clear_error()
            return
        try:
            parse_hotkey_combo(text)
        except ValueError as exc:
            self._show_error(str(exc))
            return
        self._set_combination_internal(text)
        self._clear_error()

    def _set_combination_internal(self, combo: str) -> None:
        self._combination = combo
        self._line_edit.setText(combo)
        self.combination_changed.emit(combo)

    def _show_error(self, message: str) -> None:
        self._validation_error = message
        self._error_label.setText(f"Invalid combination: {message}")
        self._error_label.setVisible(True)

    def _clear_error(self) -> None:
        self._validation_error = ""
        self._error_label.setText("")
        self._error_label.setVisible(False)

    def _key_name_for_raw(self, event: QKeyEvent) -> str | None:
        modifier = _MODIFIER_KEYS.get(event.key())
        if modifier is not None:
            return modifier
        named = _NAMED_KEYS.get(event.key())
        if named is not None:
            return named
        key = event.key()
        if _KEY_A <= key <= _KEY_Z:
            return chr(key).lower()
        if _KEY_0 <= key <= _KEY_9:
            return chr(key)
        return None
