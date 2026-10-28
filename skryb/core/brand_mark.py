"""The Skryb mark: three rising sound bars feeding a text cursor.

One set of shapes on a 256 grid draws every copy of the mark: the app
icon files, the README logo and the animated tray icon. The Settings
window draws the same grid in ``ui/src/components/mark.tsx``.
"""

import math
from dataclasses import dataclass

import numpy
from PIL import Image, ImageDraw

GRID = 256
TILE_RADIUS = 58
# Left edge and height of each bar, quietest first; all centred on y=128.
BARS = ((56, 40), (86, 72), (116, 108))
BAR_WIDTH = 18
CARET_CENTRE = 176
CARET_TOP = 58
CARET_HEIGHT = 140
CARET_STEM = 18
CARET_SERIF = 48
CARET_CAP = 14
_MIDDLE = GRID / 2

EMERALD_LIGHT = (52, 211, 153)
EMERALD_DEEP = (4, 120, 87)
WHITE = (255, 255, 255)

_SUPERSAMPLE = 4


@dataclass(frozen=True)
class Pose:
    """One frame of the mark: each bar's height scale, sideways shift and opacity."""

    scales: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shifts: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alphas: tuple[float, float, float] = (1.0, 1.0, 1.0)
    caret_alpha: float = 1.0


STILL = Pose()


def render_mark(
    size: int,
    ink: tuple[int, int, int] = WHITE,
    *,
    tile: bool = True,
    pose: Pose = STILL,
    zoom: float = 1.0,
) -> Image.Image:
    """The mark at ``size`` pixels, on the emerald tile or on transparency.

    ``zoom`` scales the mark about the centre; without the tile the mark
    can fill more of a small tray square.
    """
    big = size * _SUPERSAMPLE
    image = _tile(big) if tile else Image.new("RGBA", (big, big), (0, 0, 0, 0))
    unit = big / GRID * zoom
    offset = big / 2 - _MIDDLE * unit

    def at(value: float) -> float:
        return offset + value * unit

    for (left, height), scale, shift, alpha in zip(BARS, pose.scales, pose.shifts, pose.alphas):
        drawn = height * scale
        box = (at(left + shift), at(_MIDDLE - drawn / 2), at(left + shift + BAR_WIDTH), at(_MIDDLE + drawn / 2))
        _composite(image, alpha, lambda draw, fill, box=box: draw.rounded_rectangle(
            box, radius=BAR_WIDTH * unit / 2, fill=fill,
        ), ink)
    caret = [(at(x), at(y)) for x, y in caret_outline()]
    _composite(image, pose.caret_alpha, lambda draw, fill: draw.polygon(caret, fill=fill), ink)
    return image.resize((size, size), Image.Resampling.LANCZOS)


def caret_outline() -> list[tuple[float, float]]:
    """The I-beam cursor as a polygon on the 256 grid, clockwise from top left."""
    left = CARET_CENTRE - CARET_SERIF / 2
    right = CARET_CENTRE + CARET_SERIF / 2
    stem_left = CARET_CENTRE - CARET_STEM / 2
    stem_right = CARET_CENTRE + CARET_STEM / 2
    top = CARET_TOP
    bottom = CARET_TOP + CARET_HEIGHT
    return [
        (left, top), (right, top), (right, top + CARET_CAP), (stem_right, top + CARET_CAP),
        (stem_right, bottom - CARET_CAP), (right, bottom - CARET_CAP), (right, bottom),
        (left, bottom), (left, bottom - CARET_CAP), (stem_left, bottom - CARET_CAP),
        (stem_left, top + CARET_CAP), (left, top + CARET_CAP),
    ]


def recording_pose(t: float) -> Pose:
    """Bars bouncing, ``t`` from 0 to 1 over one loop."""
    cycles = (2, 1, 2)
    starts = (0.0, 0.3, 0.6)
    scales = tuple(
        0.3 + 0.7 * (0.5 - 0.5 * math.cos(2 * math.pi * (t * n + start)))
        for n, start in zip(cycles, starts)
    )
    return Pose(scales=scales)


def transcribing_pose(t: float) -> Pose:
    """Bars sliding into the cursor and fading as they reach it, one after another."""
    shifts, alphas, scales = [], [], []
    for index in range(len(BARS)):
        u = (t + index / len(BARS)) % 1.0
        shifts.append(-20 + 70 * u)
        alphas.append(min(u / 0.3, 1.0) * (1.0 - u))
        scales.append(1.0 - 0.6 * u)
    return Pose(scales=tuple(scales), shifts=tuple(shifts), alphas=tuple(alphas))


def loading_pose(t: float) -> Pose:
    """A soft highlight sweeping left to right across the bars and the cursor."""

    def glow(delay: float) -> float:
        return 0.28 + 0.72 * max(0.0, math.cos(2 * math.pi * (t - delay))) ** 2

    return Pose(alphas=(glow(0.0), glow(0.15), glow(0.3)), caret_alpha=glow(0.45))


def svg(*, tile: bool = True, ink: str = "#FFFFFF") -> str:
    """The still mark as SVG markup on the 256 grid."""
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {GRID} {GRID}" width="{GRID}" height="{GRID}">']
    if tile:
        light, deep = (_hex(EMERALD_LIGHT), _hex(EMERALD_DEEP))
        parts.append(
            '<defs><linearGradient id="skryb-tile" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{light}"/><stop offset="1" stop-color="{deep}"/>'
            "</linearGradient></defs>"
            f'<rect width="{GRID}" height="{GRID}" rx="{TILE_RADIUS}" fill="url(#skryb-tile)"/>'
        )
    for left, height in BARS:
        parts.append(
            f'<rect x="{left}" y="{_MIDDLE - height / 2:g}" width="{BAR_WIDTH}" height="{height}" '
            f'rx="{BAR_WIDTH / 2:g}" fill="{ink}"/>'
        )
    points = " ".join(f"{x:g},{y:g}" for x, y in caret_outline())
    parts.append(f'<polygon points="{points}" fill="{ink}"/>')
    parts.append("</svg>")
    return "".join(parts) + "\n"


def _tile(big: int) -> Image.Image:
    """The emerald squircle, its gradient running from top left to bottom right."""
    ramp = numpy.add.outer(numpy.arange(big), numpy.arange(big)) / max(2 * (big - 1), 1)
    light = numpy.array(EMERALD_LIGHT, dtype=numpy.float32)
    deep = numpy.array(EMERALD_DEEP, dtype=numpy.float32)
    rgb = light + (deep - light) * ramp[:, :, None]
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, big - 1, big - 1), radius=TILE_RADIUS * big / GRID, fill=255)
    tile = Image.fromarray(numpy.rint(rgb).astype(numpy.uint8), "RGB").convert("RGBA")
    tile.putalpha(mask)
    return tile


def _composite(image: Image.Image, alpha: float, shape, ink: tuple[int, int, int]) -> None:
    """Draw one shape on its own layer and blend it in.

    ImageDraw replaces pixels rather than blending them, so a shape drawn
    part-transparent straight onto the tile would punch a hole through it.
    """
    if alpha <= 0.004:
        return
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shape(ImageDraw.Draw(layer), (*ink, round(255 * min(alpha, 1.0))))
    image.alpha_composite(layer)


def _hex(colour: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*colour)
