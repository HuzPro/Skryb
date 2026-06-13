from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from voicetotext.settings.transcription_mode import TranscriptionMode


_MODE_HINTS: dict[TranscriptionMode, str] = {
    TranscriptionMode.BATCH: (
        "Records your dictation, then transcribes it when you stop. "
        "All batch models are available below."
    ),
    TranscriptionMode.STREAMING: (
        "Types text live while you speak (GPU only). "
        "The list below shows only streaming models."
    ),
}


class ModeSelector(QWidget):
    """Inline batch/streaming toggle shown directly above the model list.

    Emits :attr:`mode_changed` only on a *user* toggle so the enclosing
    page can re-filter the model list live; programmatic :meth:`set_mode`
    is silent. A hint line under the buttons explains what the active
    mode does and why the model list below it is filtered.
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addLayout(self._build_button_row())
        self._hint = QLabel("")
        self._hint.setObjectName("mode-hint")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

    def _build_button_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(16)
        for mode in TranscriptionMode:
            button = QRadioButton(mode.display_label)
            self._group.addButton(button)
            button.toggled.connect(self._on_toggled)
            row.addWidget(button)
            self._buttons[mode] = button
        row.addStretch(1)
        return row

    def _on_toggled(self, checked: bool) -> None:
        if not checked:
            return
        self._refresh_hint()
        if not self._suppress_emit:
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
        self._refresh_hint()

    def disable_streaming(self, reason: str) -> None:
        """Disable the Streaming option (no compatible GPU) and tooltip *reason*.

        Leaves Batch selectable. The enclosing page calls this when the
        detected hardware cannot run a streaming model, so the user can't
        toggle into a mode that would fail to load.
        """
        button = self._buttons[TranscriptionMode.STREAMING]
        button.setEnabled(False)
        button.setToolTip(reason)

    def hint_text(self) -> str:
        return self._hint.text()

    def _refresh_hint(self) -> None:
        self._hint.setText(_MODE_HINTS[self.current_mode()])
