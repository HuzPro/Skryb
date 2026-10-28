"""Write every icon file from the mark's shapes in ``skryb/core/brand_mark.py``.

    python scripts/build_icons.py

Writes ``brand/`` (the SVG, the multi-size .ico and the README wordmarks)
and the Settings window's ``ui/src-tauri/icons/icon.ico``. The outputs are
committed; run this again after changing the mark.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skryb.core.brand_mark import render_mark, svg

_REPO = Path(__file__).resolve().parent.parent
_BRAND = _REPO / "brand"
_TAURI_ICON = _REPO / "ui" / "src-tauri" / "icons" / "icon.ico"
# Every size Windows asks an .ico for, from the tray up to the Start menu's large tiles.
_ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
_WORDMARK_TEXT = {"light": "#16161A", "dark": "#F4F4F6"}


def wordmark(text_colour: str) -> str:
    """The mark beside the name, for the top of the README."""
    shapes = svg().strip().split(">", 1)[1].removesuffix("</svg>")
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 330 96" width="330" height="96">'
        f'<g transform="scale(0.375)">{shapes}</g>'
        f'<text x="118" y="68" fill="{text_colour}" font-family="Segoe UI, Inter, Helvetica, Arial, sans-serif" '
        'font-size="60" font-weight="700" letter-spacing="-1">Skryb</text></svg>\n'
    )


def write_ico(path: Path) -> None:
    # Each size drawn on its own so small ones stay sharp instead of being shrunk from 256.
    images = [render_mark(size) for size in _ICO_SIZES]
    images[-1].save(path, format="ICO", sizes=[(s, s) for s in _ICO_SIZES], append_images=images[:-1])


def main() -> None:
    _BRAND.mkdir(exist_ok=True)
    (_BRAND / "skryb.svg").write_text(svg(), encoding="utf-8")
    for scheme, colour in _WORDMARK_TEXT.items():
        (_BRAND / f"skryb-wordmark-{scheme}.svg").write_text(wordmark(colour), encoding="utf-8")
    write_ico(_BRAND / "skryb.ico")
    write_ico(_TAURI_ICON)
    print(f"Wrote {_BRAND} and {_TAURI_ICON}")


if __name__ == "__main__":
    main()
