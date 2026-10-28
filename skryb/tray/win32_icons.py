from collections.abc import Callable
from functools import lru_cache

from PIL import Image, ImageColor

from skryb.core.brand_mark import Pose, loading_pose, recording_pose, render_mark, transcribing_pose
from skryb.core.color_scheme import (
    ColorScheme,
    color_scheme_from_registry,
    detect_color_scheme,
)
from skryb.core.state_machine import AppState

# Re-exported so existing tray callers/tests keep importing the scheme and
# detector from here; the canonical home is ``core.color_scheme``.
__all__ = [
    "ColorScheme",
    "color_scheme_from_registry",
    "detect_color_scheme",
    "FRAME_MS",
    "create_icon_frames",
    "render_frames_for_scheme",
]


# Per-scheme palettes. The light palette uses darker hues so the mark
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

# How long each frame of a moving tray icon shows, and how many frames make
# one loop of each state's motion. Idle and error stand still.
FRAME_MS = 80
_MOTION: dict[AppState, tuple[Callable[[float], Pose], int]] = {
    AppState.RECORDING: (recording_pose, 15),
    AppState.TRANSCRIBING: (transcribing_pose, 16),
    AppState.LOADING: (loading_pose, 16),
}
# No tile in the tray, so the mark is drawn larger to fill the small square.
_TRAY_ZOOM = 1.6


def create_icon_frames(state: AppState, size: int = 64) -> tuple[Image.Image, ...]:
    """The tray mark for ``state`` in the detected system colour scheme:
    one image when it stands still, one loop of frames when it moves."""
    return render_frames_for_scheme(state, detect_color_scheme(), size)


@lru_cache(maxsize=32)
def render_frames_for_scheme(
    state: AppState, scheme: ColorScheme, size: int = 64,
) -> tuple[Image.Image, ...]:
    """Render the state's frames for an explicit colour scheme.

    Separated from :func:`create_icon_frames` so tests can verify palette
    routing without depending on the host's OS theme.
    """
    palette = _DARK_PALETTE if scheme == ColorScheme.DARK else _LIGHT_PALETTE
    ink = ImageColor.getrgb(palette.get(state, palette[AppState.IDLE]))[:3]
    if state not in _MOTION:
        return (render_mark(size, ink, tile=False, zoom=_TRAY_ZOOM),)
    pose, count = _MOTION[state]
    return tuple(
        render_mark(size, ink, tile=False, pose=pose(index / count), zoom=_TRAY_ZOOM)
        for index in range(count)
    )
