from PIL import Image, ImageDraw

from voicetotext.core.state_machine import AppState

_GRAY = "#808080"
_RED = "#FF3333"
_YELLOW = "#FFCC00"
_ORANGE = "#FF6600"

_COLORS = {
    AppState.IDLE: _GRAY,
    AppState.RECORDING: _RED,
    AppState.TRANSCRIBING: _YELLOW,
    AppState.ERROR: _ORANGE,
}


def create_icon(state: AppState, size: int = 64) -> Image.Image:
    color = _COLORS.get(state, "#808080")
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = size // 8
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=color,
    )
    return image
