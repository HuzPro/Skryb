from PySide6.QtWidgets import QButtonGroup, QLabel, QRadioButton, QVBoxLayout, QWidget

from voicetotext.output.output_mode import OutputMode


class OutputPage(QWidget):
    """Page for choosing how transcribed text reaches the focused window.

    Renders one radio button per :class:`OutputMode`, tagged with the
    enum value so the page can map a click back to the canonical mode.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._buttons: dict[OutputMode, QRadioButton] = {}
        self._group = QButtonGroup(self)
        self._build()
        self.set_mode(OutputMode.DIRECT_PASTE)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        heading = QLabel("Output Mode")
        heading.setObjectName("page-heading")
        layout.addWidget(heading)

        for mode in OutputMode:
            button = QRadioButton(mode.display_label)
            button.setProperty("output_mode_value", mode.value)
            self._group.addButton(button)
            layout.addWidget(button)
            self._buttons[mode] = button

        layout.addStretch(1)

    def current_mode(self) -> OutputMode:
        for mode, button in self._buttons.items():
            if button.isChecked():
                return mode
        return OutputMode.DIRECT_PASTE

    def set_mode(self, mode: OutputMode) -> None:
        button = self._buttons[mode]
        button.setChecked(True)

    def radio_buttons(self) -> list[QRadioButton]:
        return list(self._buttons.values())
