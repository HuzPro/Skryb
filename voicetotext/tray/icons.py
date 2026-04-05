from PIL import Image, ImageDraw

from voicetotext.core.state_machine import AppState

_COLORS = {
    AppState.IDLE: "#808080",
    AppState.RECORDING: "#FF3333",
    AppState.TRANSCRIBING: "#FFCC00",
    AppState.ERROR: "#FF6600",
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
