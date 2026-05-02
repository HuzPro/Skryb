from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from voicetotext.settings.qt_hotkey_capture import CaptureBridge, QtHotkeyCapture


_TOGGLE_MODE = "toggle"
_PUSH_TO_TALK_MODE = "push_to_talk"
_ALLOWED_MODES = {_TOGGLE_MODE, _PUSH_TO_TALK_MODE}


class HotkeysPage(QWidget):
    """Page for configuring the global hotkey combination and activation mode.

    Combines :class:`QtHotkeyCapture` for real key-chord capture with a
    radio group for toggle / push-to-talk. Exposes simple getters the
    main window uses to build a :class:`SettingsDraft`.
    """

    recording_state_changed = Signal(bool)

    def __init__(
        self,
        parent: QWidget | None = None,
        capture_bridge: CaptureBridge | None = None,
    ) -> None:
        super().__init__(parent)
        self._combination: str = ""
        self._mode: str = _TOGGLE_MODE
        self._mode_buttons: dict[str, QRadioButton] = {}
        self._capture_bridge = capture_bridge
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        layout.addWidget(self._make_heading("Hotkey Combination"))
        self._capture = QtHotkeyCapture(capture_bridge=self._capture_bridge)
        self._capture.combination_changed.connect(self._on_combination_changed)
        self._capture.recording_started.connect(self._on_recording_started)
        self._capture.recording_stopped.connect(self._on_recording_stopped)
        layout.addWidget(self._capture)

        hint = QLabel(
            "Type a combination, or click Record and press your keys, then click Stop."
        )
        hint.setObjectName("page-hint")
        layout.addWidget(hint)

        layout.addSpacing(12)
        layout.addWidget(self._make_heading("Activation Mode"))

        self._mode_group = QButtonGroup(self)
        self._add_mode_radio(
            layout, _TOGGLE_MODE, "Toggle (press to start, press again to stop)"
        )
        self._add_mode_radio(
            layout,
            _PUSH_TO_TALK_MODE,
            "Push-to-talk (hold to record, release to stop)",
        )

        self.set_mode(_TOGGLE_MODE)
        layout.addStretch(1)

    def _make_heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("page-heading")
        return label

    def _add_mode_radio(
        self, layout: QVBoxLayout, value: str, label: str
    ) -> None:
        button = QRadioButton(label)
        button.toggled.connect(
            lambda checked, v=value: self._on_mode_toggled(checked, v)
        )
        self._mode_group.addButton(button)
        layout.addWidget(button)
        self._mode_buttons[value] = button

    def _on_mode_toggled(self, checked: bool, value: str) -> None:
        if checked:
            self._mode = value

    def _on_combination_changed(self, combo: str) -> None:
        self._combination = combo

    def _on_recording_started(self) -> None:
        self.recording_state_changed.emit(True)

    def _on_recording_stopped(self) -> None:
        self.recording_state_changed.emit(False)

    def hotkey_capture(self) -> QtHotkeyCapture:
        return self._capture

    def current_combination(self) -> str:
        return self._combination

    def set_combination(self, combo: str) -> None:
        self._combination = combo
        self._capture.set_combination(combo)

    def current_mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode not in _ALLOWED_MODES:
            return
        self._mode = mode
        self._mode_buttons[mode].setChecked(True)
