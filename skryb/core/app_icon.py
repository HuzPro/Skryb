"""The installed app's icon, drawn in the colourway picked in Settings.

Setup copies the emerald ``skryb.ico`` beside the bundled Python, and the
Start menu and desktop shortcuts (and so a taskbar pin) show that file.
Repainting it is how they follow the chosen colour. An update copies the
emerald one back, and the next start repaints it.
"""

import ctypes
import io
import logging
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path

from skryb.core.brand_mark import TILE_GRADIENTS, render_mark
from skryb.core.release_update import is_installed_copy

_logger = logging.getLogger(__name__)

ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 256)
_SHCNE_ASSOCCHANGED = 0x08000000
_SHCNF_IDLIST = 0x0000
_painting = threading.Lock()


def paint_app_icon(colourway: str, prefix: Path = Path(sys.prefix)) -> bool:
    """Draw ``{app}\\skryb.ico`` in ``colourway``; True when the file changed.

    Only a copy Setup installed: a source checkout's ``brand/skryb.ico`` is never touched.
    """
    if not is_installed_copy(prefix):
        return False
    try:
        return _repaint_icon_file(prefix.parent / "skryb.ico", colourway)
    except Exception:  # The icon is cosmetic: it must never stop Skryb starting or saving settings.
        _logger.warning("Could not paint the app icon %s.", colourway, exc_info=True)
        return False


def paint_app_icon_in_background(colourway: Callable[[], str]) -> None:
    """Repaint on a daemon thread, in the colourway chosen when it runs, so the latest pick wins."""
    threading.Thread(target=_paint_latest, args=(colourway,), daemon=True, name="app-icon").start()


def app_icon_bytes(colourway: str) -> bytes:
    """The multi-size .ico in ``colourway``; an unknown one is green."""
    gradient = TILE_GRADIENTS.get(colourway, TILE_GRADIENTS["green"])
    # Each size drawn on its own so small ones stay sharp instead of being shrunk from 256.
    images = [render_mark(size, gradient=gradient) for size in ICO_SIZES]
    buffer = io.BytesIO()
    images[-1].save(buffer, format="ICO", sizes=[(s, s) for s in ICO_SIZES], append_images=images[:-1])
    return buffer.getvalue()


def _paint_latest(colourway: Callable[[], str]) -> None:
    with _painting:
        paint_app_icon(colourway())


def _repaint_icon_file(path: Path, colourway: str) -> bool:
    data = app_icon_bytes(colourway)
    if path.is_file() and path.read_bytes() == data:
        return False
    # A fixed name: one left behind by a failed swap is overwritten by the next paint.
    staged = path.with_name(path.name + ".tmp")
    staged.write_bytes(data)
    os.replace(staged, path)
    _tell_shell_icons_changed()
    _logger.info("Painted the app icon %s.", colourway)
    return True


def _tell_shell_icons_changed() -> None:
    """Ask Explorer to drop the icons it cached, so the shortcuts show the new file."""
    if sys.platform != "win32":
        return
    ctypes.windll.shell32.SHChangeNotify(_SHCNE_ASSOCCHANGED, _SHCNF_IDLIST, None, None)
