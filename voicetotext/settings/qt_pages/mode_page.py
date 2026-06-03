from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from voicetotext.settings.transcription_mode import TranscriptionMode


class ModePage(QWidget):
    """Page for choosing batch vs streaming transcription.

    Emits :attr:`mode_changed` only on a *user* toggle, so the enclosing
    window can re-filter the model list live; programmatic
    :meth:`set_mode` is silent.
    """

    mode_changed = Signal(object)  # emits a TranscriptionMode

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[TranscriptionMode, QRadioButton] = {}
        self._group = QButtonGroup(self)
        self._suppress_emit = False
        self._build()
        self.set_mode(TranscriptionMode.BATCH)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        heading = QLabel("Transcription Mode")
        heading.setObjectName("page-heading")
        layout.addWidget(heading)
        for mode in TranscriptionMode:
            button = QRadioButton(mode.display_label)
            self._group.addButton(button)
            button.toggled.connect(self._on_toggled)
            layout.addWidget(button)
            self._buttons[mode] = button
        layout.addStretch(1)

    def _on_toggled(self, checked: bool) -> None:
        if checked and not self._suppress_emit:
            self.mode_changed.emit(self.current_mode())

    def current_mode(self) -> TranscriptionMode:
        for mode, button in self._buttons.items():
            if button.isChecked():
                return mode
        return TranscriptionMode.BATCH

    def set_mode(self, mode: TranscriptionMode) -> None:
        self._suppress_emit = True
        try:
            self._buttons[mode].setChecked(True)
        finally:
            self._suppress_emit = False
