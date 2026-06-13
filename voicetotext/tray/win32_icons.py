from enum import Enum

from PIL import Image, ImageDraw

from voicetotext.core.state_machine import AppState


class ColorScheme(Enum):
    LIGHT = "light"
    DARK = "dark"


# Per-scheme palettes. The light palette uses darker hues so the circle
# stays visible on a white/grey Windows taskbar; the dark palette uses
# lighter hues for the inverse case. Both were picked for ~4.5:1 contrast
# against their target background (WCAG AA).
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

_SUPERSAMPLE = 4  # draw large then downscale: PIL's ellipse is not anti-aliased


def create_icon_image(state: AppState, size: int = 64) -> Image.Image:
    """A filled circle in the colour matching ``state`` and the detected
    system colour scheme."""
    return render_image_for_scheme(state, detect_color_scheme(), size)


def render_image_for_scheme(
    state: AppState, scheme: ColorScheme, size: int = 64,
) -> Image.Image:
    """Render the state icon for an explicit colour scheme.

    Separated from :func:`create_icon_image` so tests can verify palette
    routing without depending on the host's OS theme.
    """
    palette = _DARK_PALETTE if scheme == ColorScheme.DARK else _LIGHT_PALETTE
    color = palette.get(state, palette[AppState.IDLE])
    large = size * _SUPERSAMPLE
    image = Image.new("RGBA", (large, large), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = large / 8.0
    draw.ellipse((margin, margin, large - margin, large - margin), fill=color)
    return image.resize((size, size), Image.LANCZOS)


def detect_color_scheme() -> ColorScheme:
    """Read the Windows light/dark setting, defaulting to light on error."""
    import winreg

    return color_scheme_from_registry(winreg)


def color_scheme_from_registry(winreg: object) -> ColorScheme:
    try:
        return _read_color_scheme(winreg)
    except OSError:
        return ColorScheme.LIGHT


def _read_color_scheme(winreg: object) -> ColorScheme:
    key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        uses_light, _ = winreg.QueryValueEx(key, "SystemUsesLightTheme")
    return ColorScheme.LIGHT if uses_light else ColorScheme.DARK
