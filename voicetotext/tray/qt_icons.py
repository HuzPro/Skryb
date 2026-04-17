from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPixmap

from voicetotext.core.state_machine import AppState


_COLORS: dict[AppState, str] = {
    AppState.IDLE: "#808080",
    AppState.RECORDING: "#FF3333",
    AppState.TRANSCRIBING: "#FFCC00",
    AppState.ERROR: "#FF6600",
}
_DEFAULT_COLOR = "#808080"


def create_qicon(state: AppState, size: int = 64) -> QIcon:
    """Paint a filled circle in the colour matching ``state``."""
    color = _COLORS.get(state, _DEFAULT_COLOR)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(color)))
        margin = size / 8.0
        painter.drawEllipse(
            QRectF(margin, margin, size - 2 * margin, size - 2 * margin)
        )
    finally:
        painter.end()

    return QIcon(pixmap)
