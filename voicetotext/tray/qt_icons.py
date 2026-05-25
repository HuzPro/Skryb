from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QGuiApplication,
    QIcon,
    QPainter,
    QPixmap,
)

from voicetotext.core.state_machine import AppState


# Per-scheme palettes. The light palette uses darker hues so the
# circle stays visible on a white/grey Windows taskbar; the dark
# palette uses lighter hues for the inverse case. Both were picked
# for ~4.5:1 contrast against their target background (WCAG AA).
_LIGHT_PALETTE: dict[AppState, str] = {
    AppState.LOADING: "#1F6FEB",
    AppState.IDLE: "#4F5560",
    AppState.RECORDING: "#D63A3A",
    AppState.TRANSCRIBING: "#B07000",
    AppState.ERROR: "#B83B00",
}
_DARK_PALETTE: dict[AppState, str] = {
    AppState.LOADING: "#6FA8FF",
    AppState.IDLE: "#C9CCD0",
    AppState.RECORDING: "#FF5959",
    AppState.TRANSCRIBING: "#FFD650",
    AppState.ERROR: "#FF8A3D",
}


def create_qicon(state: AppState, size: int = 64) -> QIcon:
    """Paint a filled circle in the colour matching ``state`` and the
    detected system colour scheme."""
    return render_qicon_for_scheme(state, detect_color_scheme(), size)


def render_qicon_for_scheme(
    state: AppState, scheme: Qt.ColorScheme, size: int = 64,
) -> QIcon:
    """Render the state icon for an explicit colour scheme.

    Separated from :func:`create_qicon` so tests can verify palette
    routing without standing up a Qt application that mimics a
    particular OS theme.
    """
    palette = _DARK_PALETTE if scheme == Qt.ColorScheme.Dark else _LIGHT_PALETTE
    color = palette.get(state, palette[AppState.IDLE])
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


def detect_color_scheme() -> Qt.ColorScheme:
    """Read the system colour scheme via Qt's style hints.

    Falls back to ``Qt.ColorScheme.Light`` when no Qt application
    exists yet (icon creation can happen during bootstrap before
    tray init) or the platform reports ``Unknown``.
    """
    app = QGuiApplication.instance()
    if app is None:
        return Qt.ColorScheme.Light
    scheme = app.styleHints().colorScheme()
    if scheme == Qt.ColorScheme.Unknown:
        return Qt.ColorScheme.Light
    return scheme
