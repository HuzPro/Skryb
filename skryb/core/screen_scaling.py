"""Drawing at the screen's own resolution when Windows scales the display, 150% on most laptops."""

import ctypes
import logging
import sys
from functools import cache

_logger = logging.getLogger(__name__)

_PROCESS_SYSTEM_DPI_AWARE = 1
_BASE_DPI = 96


def become_dpi_aware() -> None:
    """Tell Windows Skryb scales its own windows; call before the first window exists.

    Otherwise Windows draws the tray menu, the pill and the notice at 100% and stretches them, which blurs
    them. System-aware, as IDLE is: on a second monitor with another scaling Windows still stretches them.
    """
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(_PROCESS_SYSTEM_DPI_AWARE)
    except (AttributeError, OSError) as exc:
        _logger.info("Windows stretches Skryb's windows itself: %s", exc)


@cache
def display_scale() -> float:
    """Screen pixels per pixel of the pill's design: 1.5 at 150%, 1.0 while Windows stretches it instead."""
    if sys.platform != "win32":
        return 1.0
    try:
        return ctypes.windll.user32.GetDpiForSystem() / _BASE_DPI
    except (AttributeError, OSError):
        return 1.0


def scaled_size(width: int, height: int, scale: float) -> tuple[int, int]:
    return round(width * scale), round(height * scale)
