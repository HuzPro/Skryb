from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


_MODIFIER_ONLY_KEYS = {
    Qt.Key_Control,
    Qt.Key_Alt,
    Qt.Key_AltGr,
    Qt.Key_Shift,
    Qt.Key_Meta,
    Qt.Key_Super_L,
    Qt.Key_Super_R,
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


class QtHotkeyCapture(QWidget):
    """Widget that captures a keyboard combination by recording actual key presses.

    Beats free-text entry: the user clicks 'Record', presses the combo,
    and the widget parses the modifiers + key into the canonical
    ``ctrl+alt+m`` format understood by
    :func:`voicetotext.hotkeys.key_parser.parse_hotkey_combo`.

    Escape cancels and keeps the previous value. Holding only modifier
    keys does not commit — the widget waits for a non-modifier to
    complete the combo.
    """

    combination_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._combination: str = ""
        self._recording: bool = False

        self._display = QLabel("<none>")
        self._display.setObjectName("hotkey-display")
        self._record_button = QPushButton("Record")
        self._record_button.setCheckable(True)
        self._record_button.toggled.connect(self._on_record_toggled)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._display, stretch=1)
        layout.addWidget(self._record_button)

        self.setFocusPolicy(Qt.StrongFocus)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def combination(self) -> str:
        return self._combination

    def set_combination(self, combo: str) -> None:
        self._combination = combo
        self._refresh_display()

    def is_recording(self) -> bool:
        return self._recording

    def start_recording(self) -> None:
        if not self._record_button.isChecked():
            self._record_button.setChecked(True)
        else:
            self._on_record_toggled(True)

    # ------------------------------------------------------------------
    # Qt event handling
    # ------------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._recording:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key_Escape:
            self._stop_recording()
            return
        combo = self._combo_from_event(event)
        if combo is None:
            return
        self._commit_combo(combo)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_record_toggled(self, checked: bool) -> None:
        self._recording = checked
        if checked:
            self._display.setText("press keys…")
            self.setFocus()
        else:
            self._refresh_display()

    def _commit_combo(self, combo: str) -> None:
        self._combination = combo
        self.combination_changed.emit(combo)
        self._stop_recording()

    def _stop_recording(self) -> None:
        self._record_button.setChecked(False)

    def _refresh_display(self) -> None:
        self._display.setText(self._combination if self._combination else "<none>")

    def _combo_from_event(self, event: QKeyEvent) -> str | None:
        if event.key() in _MODIFIER_ONLY_KEYS:
            return None
        key_name = self._key_name(event)
        if key_name is None:
            return None
        parts: list[str] = []
        modifiers = event.modifiers()
        if modifiers & Qt.ControlModifier:
            parts.append("ctrl")
        if modifiers & Qt.AltModifier:
            parts.append("alt")
        if modifiers & Qt.ShiftModifier:
            parts.append("shift")
        if modifiers & Qt.MetaModifier:
            parts.append("cmd")
        parts.append(key_name)
        return "+".join(parts)

    def _key_name(self, event: QKeyEvent) -> str | None:
        named = _NAMED_KEYS.get(event.key())
        if named is not None:
            return named
        text = event.text().lower()
        if len(text) == 1 and text.isalnum():
            return text
        return None
