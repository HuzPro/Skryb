"""Shape, motion and drawing of the recording pill. No Win32 and no Tk here.

The pill lives on a fixed transparent canvas and only its drawn shape
changes size, so the window never moves or resizes mid-animation and the
transparent margin passes clicks through to whatever is underneath.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache, lru_cache

import numpy
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from skryb.core.brand_mark import EMERALD_DEEP, EMERALD_LIGHT
from skryb.core.screen_scaling import scaled_size

CANVAS_WIDTH = 248
CANVAS_HEIGHT = 72
# The pill's bottom edge stays put and it grows upward, so the canvas keeps
# a little room below it for the overshoot of the spring.
_BOTTOM_INSET = 10
_SUPERSAMPLE = 3

# Where the pill sits, as how far along its travel it is across and down
# the work area: 0.5 across is centred, 1.0 down is as low as it goes,
# which keeps the margin above the taskbar it has always had.
PillPosition = tuple[float, float]
DEFAULT_PILL_POSITION: PillPosition = (0.5, 1.0)
BOTTOM_MARGIN = 26
# Fractions come back from a division; the last bit of float error is
# forgiven before flooring back to a whole pixel.
_PIXEL_SLACK = 1e-6

_RECORDING_SIZE = (220.0, 48.0)
# One pill: the progress bar takes over the recording pill's exact shape
# and colour, so nothing jumps when the recording stops.
_TRANSCRIBING_SIZE = _RECORDING_SIZE
_COLLAPSED_SIZE = (44.0, 10.0)

_PADDING = 8.0
_BUTTON_GAP = 6.0
_WAVE_MARGIN = 12.0
_BAR_COUNT = 11
_BAR_WIDTH = 3.0
_BAR_GAP = 4.0
_BAR_MIN_HEIGHT = 3.0

_LABEL_TEXT = "Transcribing"
_LABEL_SIZE = 12
_NOTICE_TEXT = "No audio from"
_NOTICE_SIZE = 11
_NOTICE_LINE_OFFSET = 7.0
_TEXT_MARGIN = 6.0
_COUNTDOWN_SIZE = 12
_COUNTDOWN_GAP = 6.0
_COUNTDOWN_BAR_COUNT = 7

# A little under critical damping, so the pill lands with one small
# bounce the way Typeless's does, then settles.
_STIFFNESS = 380.0
_DAMPING = 26.0
_SPRING_STEP_SECONDS = 1 / 240


Rgb = tuple[int, int, int]


@dataclass(frozen=True)
class PillSkin:
    """The pill's and the notice's colours and corners in one theme, light or dark.

    ``light`` skins have a pale fill, so the transcribing shimmer darkens it
    towards ``shimmer`` instead of lifting it. ``roundness`` is the corner
    radius as a share of half the height: 1 is a capsule. ``glass`` skins
    tint a frosted copy of what is behind them, catch the light along their
    rim and cast a soft shadow.
    """

    fill: Rgb
    fill_alpha: int
    border: tuple[int, int, int, int]
    button: Rgb
    button_hover: Rgb
    finish: Rgb
    finish_hover: Rgb
    glyph: Rgb
    finish_glyph: Rgb
    label: Rgb
    warning: Rgb
    wave: Rgb
    shimmer: Rgb
    light: bool = False
    roundness: float = 1.0
    glass: bool = False


# Each theme's colours come from its tokens in ui/src/index.css: the card
# or popover for the fill, the primary for the finish button and the bars,
# the warning for a silent microphone. The default dark pill is the one the
# app has always had, with the brand emerald for its bars and shimmer.
DEFAULT_SKIN = PillSkin(
    fill=(12, 12, 14), fill_alpha=250, border=(255, 255, 255, 26),
    button=(58, 58, 62), button_hover=(82, 82, 88),
    finish=(255, 255, 255), finish_hover=(222, 222, 226),
    glyph=(255, 255, 255), finish_glyph=(10, 10, 12), label=(236, 236, 240),
    warning=(251, 191, 36), wave=EMERALD_LIGHT, shimmer=EMERALD_DEEP,
)
_SKINS: dict[tuple[str, bool], PillSkin] = {
    ("default", True): DEFAULT_SKIN,
    ("default", False): PillSkin(
        fill=(255, 255, 255), fill_alpha=250, border=(0, 0, 0, 30),
        button=(241, 241, 243), button_hover=(226, 226, 230),
        finish=(24, 24, 27), finish_hover=(63, 63, 70),
        glyph=(24, 24, 27), finish_glyph=(255, 255, 255), label=(24, 24, 27),
        warning=(180, 83, 9), wave=EMERALD_DEEP, shimmer=(167, 243, 208), light=True,
    ),
    # Glass: a neutral tint over a frosted copy of what is behind (frosted()).
    ("glass", True): PillSkin(
        fill=(28, 28, 30), fill_alpha=125, border=(255, 255, 255, 46),
        button=(72, 72, 76), button_hover=(94, 94, 100),
        finish=(78, 161, 255), finish_hover=(122, 184, 255),
        glyph=(245, 245, 247), finish_glyph=(4, 17, 31), label=(245, 245, 247),
        warning=(255, 194, 97), wave=(78, 161, 255), shimmer=(52, 84, 128), glass=True,
    ),
    ("glass", False): PillSkin(
        fill=(250, 250, 252), fill_alpha=150, border=(0, 0, 0, 34),
        button=(234, 234, 238), button_hover=(220, 220, 226),
        finish=(0, 102, 204), finish_hover=(0, 88, 176),
        glyph=(29, 29, 31), finish_glyph=(255, 255, 255), label=(29, 29, 31),
        warning=(180, 83, 9), wave=(0, 102, 204), shimmer=(200, 222, 246), light=True, glass=True,
    ),
    # Material: tonal surfaces and large, not round, corners.
    ("material", True): PillSkin(
        fill=(33, 35, 47), fill_alpha=250, border=(0, 0, 0, 0),
        button=(53, 57, 97), button_hover=(67, 72, 120),
        finish=(174, 184, 254), finish_hover=(196, 205, 255),
        glyph=(226, 228, 237), finish_glyph=(35, 33, 104), label=(226, 228, 237),
        warning=(238, 186, 112), wave=(174, 184, 254), shimmer=(77, 83, 151), roundness=0.67,
    ),
    ("material", False): PillSkin(
        fill=(230, 233, 247), fill_alpha=250, border=(0, 0, 0, 0),
        button=(220, 224, 246), button_hover=(209, 216, 255),
        finish=(81, 82, 193), finish_hover=(96, 100, 213),
        glyph=(24, 26, 36), finish_glyph=(255, 255, 255), label=(24, 26, 36),
        warning=(174, 103, 0), wave=(81, 82, 193), shimmer=(188, 196, 255), light=True, roundness=0.67,
    ),
    # Paper: ink on cream inside a ruled edge, with a card's corners, in a
    # colourway's accent (see _PAPER_ACCENTS); these are the terracotta.
    ("paper", True): PillSkin(
        fill=(35, 32, 28), fill_alpha=250, border=(68, 61, 52, 255),
        button=(52, 47, 41), button_hover=(68, 61, 52),
        finish=(224, 144, 107), finish_hover=(232, 165, 133),
        glyph=(236, 229, 216), finish_glyph=(29, 20, 16), label=(236, 229, 216),
        warning=(231, 178, 100), wave=(224, 144, 107), shimmer=(86, 58, 44), roundness=0.5,
    ),
    ("paper", False): PillSkin(
        fill=(255, 253, 248), fill_alpha=250, border=(221, 212, 194, 255),
        button=(242, 237, 227), button_hover=(228, 220, 203),
        finish=(154, 75, 44), finish_hover=(176, 88, 58),
        glyph=(29, 26, 22), finish_glyph=(255, 250, 243), label=(29, 26, 22),
        warning=(154, 91, 16), wave=(154, 75, 44), shimmer=(240, 214, 196), light=True, roundness=0.5,
    ),
}


# Paper's colourways, the Skryb icon's green first: the accent and the text
# on it, light then dark. The window's twin is in ui/src/index.css.
PAPER_COLOURWAYS = ("green", "terracotta", "ink", "plum", "graphite")
_PAPER_ACCENTS: dict[str, tuple[tuple[Rgb, Rgb], tuple[Rgb, Rgb]]] = {
    "green": ((EMERALD_DEEP, (255, 255, 255)), (EMERALD_LIGHT, (4, 36, 26))),
    "terracotta": (((154, 75, 44), (255, 250, 243)), ((224, 144, 107), (29, 20, 16))),
    "ink": (((42, 82, 136), (255, 255, 255)), ((143, 181, 232), (12, 26, 44))),
    "plum": (((125, 58, 112), (255, 255, 255)), ((216, 155, 200), (42, 15, 36))),
    "graphite": (((70, 72, 77), (255, 255, 255)), ((203, 199, 191), (27, 26, 24))),
}


# Hovering deepens the accent under white text and lifts it under dark text,
# so the text only gets easier to read.
_AWAY_FROM_INK: dict[bool, Rgb] = {True: (0, 0, 0), False: (255, 255, 255)}


def pill_skin(theme: object, dark: bool, colourway: object = PAPER_COLOURWAYS[0]) -> PillSkin:
    """The pill for the window's ``theme``, light or dark; an unknown theme gets the default.

    Paper takes its accent from ``colourway``; an unknown one is the green.
    """
    key = (theme if isinstance(theme, str) else "default", dark)
    skin = _SKINS.get(key) or _SKINS[("default", dark)]
    if key[0] != "paper":
        return skin
    known = colourway if isinstance(colourway, str) and colourway in _PAPER_ACCENTS else PAPER_COLOURWAYS[0]
    accent, ink = _PAPER_ACCENTS[known][dark]
    return replace(
        skin, finish=accent, finish_hover=_mixed(accent, _AWAY_FROM_INK[skin.light], 0.14), finish_glyph=ink,
        wave=accent, shimmer=_mixed(accent, skin.fill, 0.82 if skin.light else 0.65),
    )


def without_glass(skin: PillSkin) -> PillSkin:
    """A glass skin made solid, for when Windows can't show through it; any other skin as it is."""
    return replace(skin, glass=False, fill_alpha=250) if skin.glass else skin


def _mixed(colour: Rgb, other: Rgb, share: float) -> Rgb:
    red, green, blue = (round(a + (b - a) * share) for a, b in zip(colour, other, strict=True))
    return red, green, blue


class Mode(Enum):
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    CLOSED = "closed"


class Button(Enum):
    CANCEL = "cancel"
    PAUSE = "pause"
    FINISH = "finish"


# Loudness is mapped in decibels, between the room's own noise floor and
# the speaker's recent peak. Hearing is logarithmic: on a linear scale a
# normal speaking voice (about -34 dBFS on a desk microphone) reads as a
# third of a shout and quiet speech barely lifts the bars. The floor
# drops at once to anything quieter and creeps up slowly, so pauses
# between words keep re-measuring the room; the peak jumps to anything
# louder and falls a few dB a second, so one cough does not flatten the
# next sentence.
_FLOOR_START_DB = -60.0
_FLOOR_LOWEST_DB = -75.0
_FLOOR_RISE_DB_PER_SECOND = 2.0
_GATE_ABOVE_FLOOR_DB = 8.0
_PEAK_START_DB = -30.0
_PEAK_FALL_DB_PER_SECOND = 4.0
_SMALLEST_SPAN_DB = 18.0
_LEVEL_CURVE = 0.85


class VoiceLevel:
    """Microphone loudness mapped to 0..1 between the room's noise floor and the recent peak."""

    def __init__(self) -> None:
        self._floor = _FLOOR_START_DB
        self._peak = _PEAK_START_DB

    def update(self, rms: float, seconds: float) -> float:
        if not math.isfinite(rms) or rms <= 0:
            rms = 0.0
        db = 20 * math.log10(max(rms, 1e-7))
        seconds = max(seconds, 0.0)
        self._floor = max(min(db, self._floor + _FLOOR_RISE_DB_PER_SECOND * seconds), _FLOOR_LOWEST_DB)
        self._peak = max(db, self._peak - _PEAK_FALL_DB_PER_SECOND * seconds)
        gate = self._floor + _GATE_ABOVE_FLOOR_DB
        top = max(self._peak, gate + _SMALLEST_SPAN_DB)
        return min(max((db - gate) / (top - gate), 0.0), 1.0) ** _LEVEL_CURVE


# The waveform scrolls: each bar is one slice of time, the newest on the
# right still filling as you speak, and the row slides left continuously
# so a bar never jumps between slots.
_SLICE_SECONDS = 0.075


class VoiceHistory:
    """Voice level over the last ``slices`` slices of time, oldest first."""

    def __init__(self, slices: int) -> None:
        self._voice = VoiceLevel()
        self._levels = [0.0] * slices
        self._age = 0.0

    def clear(self) -> None:
        """Flat bars for a new recording; the learnt loudness stays."""
        self._levels = [0.0] * len(self._levels)
        self._age = 0.0

    def update(self, rms: float, seconds: float) -> None:
        level = self._voice.update(rms, seconds)
        self._age += max(seconds, 0.0)
        while self._age >= _SLICE_SECONDS:
            self._age -= _SLICE_SECONDS
            self._levels = [*self._levels[1:], 0.0]
        self._levels[-1] = max(self._levels[-1], level)

    def levels(self) -> tuple[float, ...]:
        return tuple(self._levels)

    def scroll(self) -> float:
        """How far the row has slid towards the next slot, 0 to 1."""
        return self._age / _SLICE_SECONDS


class Spring:
    """A damped spring chasing ``target``; the whole motion system."""

    def __init__(
        self, value: float, stiffness: float = _STIFFNESS, damping: float = _DAMPING,
    ) -> None:
        self.value = value
        self.target = value
        self.velocity = 0.0
        self._stiffness = stiffness
        self._damping = damping

    def step(self, seconds: float) -> None:
        steps = max(1, math.ceil(seconds / _SPRING_STEP_SECONDS))
        dt = seconds / steps
        for _ in range(steps):
            pull = self._stiffness * (self.target - self.value)
            self.velocity += (pull - self._damping * self.velocity) * dt
            self.value += self.velocity * dt

    def settled(self, tolerance: float = 0.01) -> bool:
        return (
            abs(self.target - self.value) < tolerance
            and abs(self.velocity) < tolerance
        )


@dataclass(frozen=True)
class Frame:
    width: float
    height: float
    opacity: float
    # How much of the recording controls and of the transcribing label to
    # draw, 0 to 1, so each fades as the pill changes shape.
    controls: float
    label: float
    # Voice level per slice of time, oldest first, and how far the row has
    # slid towards the next slot.
    levels: tuple[float, ...] = ()
    scroll: float = 0.0
    paused: bool = False
    # Where the transcribing shimmer is across the pill, 0 to 1; below 0 for none.
    shimmer: float = -1.0
    hovered: Button | None = None
    # Shown in amber in place of the waveform while the microphone is silent.
    silent_microphone: str = ""
    # Time left before the recording limit, in amber beside the waveform.
    countdown: str = ""


def target_size(mode: Mode) -> tuple[float, float]:
    if mode is Mode.RECORDING:
        return _RECORDING_SIZE
    if mode is Mode.TRANSCRIBING:
        return _TRANSCRIBING_SIZE
    return _COLLAPSED_SIZE


def collapsed_size() -> tuple[float, float]:
    return _COLLAPSED_SIZE


def pill_origin(
    position: PillPosition, work_area: tuple[int, int, int, int],
) -> tuple[int, int]:
    """The window's top-left corner for ``position`` inside ``work_area``."""
    left, top, right, bottom = work_area
    across, down = (min(max(value, 0.0), 1.0) for value in position)
    x = left + math.floor(across * _travel(right - left, CANVAS_WIDTH) + _PIXEL_SLACK)
    y = top + math.floor(
        down * _travel(bottom - top, CANVAS_HEIGHT + BOTTOM_MARGIN) + _PIXEL_SLACK,
    )
    return x, y


def pill_position(
    origin: tuple[int, int], work_area: tuple[int, int, int, int],
) -> PillPosition:
    """Where a window at ``origin`` sits in ``work_area``; off the edge is the edge."""
    left, top, right, bottom = work_area
    return (
        _fraction(origin[0] - left, _travel(right - left, CANVAS_WIDTH), 0.5),
        _fraction(
            origin[1] - top, _travel(bottom - top, CANVAS_HEIGHT + BOTTOM_MARGIN), 1.0,
        ),
    )


def _travel(span: int, size: int) -> int:
    return max(span - size, 0)


def _fraction(offset: int, travel: int, default: float) -> float:
    if travel <= 0:
        return default
    return min(max(offset / travel, 0.0), 1.0)


def countdown_text(seconds: float) -> str:
    """``m:ss`` left, rounded up so the last second reads 0:01, not 0:00."""
    whole = max(math.ceil(seconds), 0) if math.isfinite(seconds) else 0
    return f"{whole // 60}:{whole % 60:02d}"


def pill_box(width: float, height: float) -> tuple[float, float, float, float]:
    left = (CANVAS_WIDTH - width) / 2
    bottom = CANVAS_HEIGHT - _BOTTOM_INSET
    return left, bottom - height, left + width, bottom


def button_centres(frame: Frame) -> dict[Button, tuple[float, float, float]]:
    """Each button's centre and radius, in canvas pixels."""
    left, top, right, bottom = pill_box(frame.width, frame.height)
    radius = max((frame.height - 2 * _PADDING) / 2, 0.0)
    middle = (top + bottom) / 2
    finish_x = right - _PADDING - radius
    return {
        Button.CANCEL: (left + _PADDING + radius, middle, radius),
        Button.PAUSE: (finish_x - 2 * radius - _BUTTON_GAP, middle, radius),
        Button.FINISH: (finish_x, middle, radius),
    }


def hit_test(frame: Frame, x: float, y: float) -> Button | None:
    if frame.controls < 0.9:
        return None
    for button, (cx, cy, radius) in button_centres(frame).items():
        # A couple of pixels of slack: the circles are small targets.
        if math.hypot(x - cx, y - cy) <= radius + 2:
            return button
    return None


# While transcribing, a soft band of the accent colour sweeps across the
# pill, easing in and out of each pass. On a dark pill it only lifts the
# fill towards the band colour, never darkens anything, so the white label
# reads the same throughout; on a light one it only darkens, so the dark
# label does. Applied to finished pixels with numpy: a full redraw costs
# milliseconds, and a frame at 240 Hz has four.
_SHIMMER_SECONDS = 0.9
_SHIMMER_WIDTH = 130.0
_SHIMMER_STRENGTH = 0.6


def shimmer_position(seconds: float) -> float:
    """Where the band is ``seconds`` into transcribing: 0 to 1, repeating."""
    t = (max(seconds, 0.0) / _SHIMMER_SECONDS) % 1.0
    return t * t * (3 - 2 * t)


def shimmered(pixels: numpy.ndarray, frame: Frame, skin: PillSkin = DEFAULT_SKIN) -> numpy.ndarray:
    """``pixels`` (premultiplied BGRA) with the shimmer band drawn at ``frame.shimmer``."""
    if frame.shimmer < 0 or frame.label <= 0:
        return pixels
    left, _top, right, _bottom = pill_box(frame.width, frame.height)
    centre = left - _SHIMMER_WIDTH / 2 + (right - left + _SHIMMER_WIDTH) * frame.shimmer
    scale = pixels.shape[1] / CANVAS_WIDTH
    x = (numpy.arange(pixels.shape[1], dtype=numpy.float32) + 0.5) / scale
    distance = numpy.clip(numpy.abs(x - centre) / (_SHIMMER_WIDTH / 2), 0.0, 1.0)
    weight = (numpy.cos(distance * math.pi) * 0.5 + 0.5) * (_SHIMMER_STRENGTH * frame.label)
    columns = numpy.nonzero(weight > 0.002)[0]
    if columns.size == 0:
        return pixels
    span = slice(int(columns[0]), int(columns[-1]) + 1)
    region = pixels[:, span, :].astype(numpy.float32)
    colour = numpy.array(skin.shimmer[::-1], dtype=numpy.float32) * (region[:, :, 3:4] / 255.0)
    rgb = region[:, :, :3]
    lifted = rgb + (colour - rgb) * weight[span][None, :, None]
    region[:, :, :3] = numpy.minimum(rgb, lifted) if skin.light else numpy.maximum(rgb, lifted)
    out = pixels.copy()
    out[:, span, :] = numpy.rint(region).astype(numpy.uint8)
    return out


def settled_label_frame() -> Frame:
    """The transcribing pill at rest: drawn once, then only the shimmer moves."""
    width, height = _TRANSCRIBING_SIZE
    return Frame(width, height, opacity=1.0, controls=0.0, label=1.0)


def is_settled_label(frame: Frame) -> bool:
    width, height = _TRANSCRIBING_SIZE
    return (
        abs(frame.width - width) < 0.05
        and abs(frame.height - height) < 0.05
        and frame.opacity > 0.999
        and frame.controls < 0.001
        and frame.label > 0.999
    )


def render(frame: Frame, screen_scale: float = 1.0, skin: PillSkin = DEFAULT_SKIN) -> Image.Image:
    """The canvas with ``frame`` drawn on it in ``skin``, ``screen_scale`` screen pixels to each of the design's."""
    scale = _SUPERSAMPLE
    image = Image.new(
        "RGBA", (CANVAS_WIDTH * scale, CANVAS_HEIGHT * scale), (0, 0, 0, 0),
    )
    if frame.opacity > 0.004 and frame.width > 1 and frame.height > 1:
        _draw_pill(image, frame, skin)
    return image.resize(scaled_size(CANVAS_WIDTH, CANVAS_HEIGHT, screen_scale), Image.Resampling.LANCZOS)


def _draw_pill(image: Image.Image, frame: Frame, skin: PillSkin) -> None:
    scale = _SUPERSAMPLE
    box = tuple(edge * scale for edge in pill_box(frame.width, frame.height))
    layer = _body(image.size, box, frame.height * scale / 2 * skin.roundness, skin)
    # ImageDraw replaces pixels rather than blending them, so anything
    # drawn part-transparent would punch a hole through the pill. Each
    # fading part is drawn opaque on its own layer and blended in instead.
    if frame.label > 0:
        label = _blank(image)
        _draw_label(ImageDraw.Draw(label), box, skin)
        # The label waits for the buttons to be mostly gone, so the two
        # never sit on top of each other mid-change.
        _blend(layer, label, frame.label * max(0.0, 1.0 - 2.0 * frame.controls))
    if frame.controls > 0:
        controls = _blank(image)
        _draw_controls(ImageDraw.Draw(controls), frame, skin)
        _blend(layer, controls, frame.controls)
    _fade(layer, frame.opacity)
    image.alpha_composite(layer)


def _body(size: tuple[int, int], box: tuple[float, ...], radius: float, skin: PillSkin) -> Image.Image:
    """A new layer of ``size`` with the body drawn on it; glass bodies are kept, as they cost more to draw."""
    if skin.glass:
        return _glass_body(size, _snapped(box), round(radius), skin).copy()
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    # Drawn pixels replace those under them, so a clear border would cut a
    # ring out of the fill: a theme without one draws none.
    ImageDraw.Draw(layer).rounded_rectangle(
        box,
        radius=radius,
        fill=(*skin.fill, skin.fill_alpha),
        outline=skin.border if skin.border[3] else None,
        width=_SUPERSAMPLE,
    )
    return layer


def _snapped(box: tuple[float, ...]) -> tuple[int, int, int, int]:
    """``box`` on whole supersampled pixels, so a body that is still settling draws as the settled one."""
    left, top, right, bottom = (round(edge) for edge in box)
    return left, top, right, bottom


# Glass catches the light from above: a sheen over its top and a rim lit
# brightest along the top edge, a little along the bottom. Its shadow falls
# just below it, and is cut off where it fades, so only pixels close to the
# pill take its clicks.
_SHEEN = {True: 70, False: 34}
_RIM = {True: (255, 110, 150), False: (210, 70, 120)}
_SHADOW_DROP = 2.0
_SHADOW_BLUR = 3.0
_SHADOW_STRENGTH = 0.35
_SHADOW_FLOOR = 10


@lru_cache(maxsize=16)
def _glass_body(size: tuple[int, int], box: tuple[int, int, int, int], radius: int, skin: PillSkin) -> Image.Image:
    scale = _SUPERSAMPLE
    body = _shape(size, box, radius)
    shadow = ImageChops.offset(body, 0, round(_SHADOW_DROP * scale)).filter(
        ImageFilter.GaussianBlur(_SHADOW_BLUR * scale),
    )
    shadow = ImageChops.subtract(shadow, body).point(
        lambda value: 0 if value * _SHADOW_STRENGTH < _SHADOW_FLOOR else round(value * _SHADOW_STRENGTH),
    )
    layer = _solid(size, (0, 0, 0), shadow)
    ImageDraw.Draw(layer).rounded_rectangle(box, radius=radius, fill=(*skin.fill, skin.fill_alpha))
    inner = _shape(size, (box[0] + scale, box[1] + scale, box[2] - scale, box[3] - scale), max(radius - scale, 0))
    edge = ImageChops.subtract(body, inner)
    layer.alpha_composite(_solid(size, skin.border, ImageChops.multiply(edge, Image.new("L", size, skin.border[3]))))
    top, bottom = box[1], box[3]
    rows = numpy.clip((numpy.arange(size[1], dtype=numpy.float32) - top) / max(bottom - top, 1), 0.0, 1.0)
    sheen = numpy.clip(1.0 - rows / 0.5, 0.0, 1.0) ** 2 * _SHEEN[skin.light]
    rim = numpy.interp(rows, (0.0, 0.5, 1.0), _RIM[skin.light])
    light = ImageChops.add(
        ImageChops.multiply(inner, _rows(size, sheen)), ImageChops.multiply(edge, _rows(size, rim)),
    )
    layer.alpha_composite(_solid(size, (255, 255, 255), light))
    return layer


def _shape(size: tuple[int, int], box: tuple[int, int, int, int], radius: int) -> Image.Image:
    shape = Image.new("L", size, 0)
    ImageDraw.Draw(shape).rounded_rectangle(box, radius=radius, fill=255)
    return shape


def _rows(size: tuple[int, int], values: numpy.ndarray) -> Image.Image:
    """An 8-bit image whose rows are each one of ``values``, top to bottom."""
    column = numpy.rint(values).astype(numpy.uint8)[:, None]
    return Image.fromarray(numpy.ascontiguousarray(numpy.broadcast_to(column, (size[1], size[0]))))


def _solid(size: tuple[int, int], colour: tuple[int, ...], alpha: Image.Image) -> Image.Image:
    layer = Image.new("RGBA", size, (colour[0], colour[1], colour[2], 0))
    layer.putalpha(alpha)
    return layer


# Through glass, what is behind is blurred, and near the rim it bends the
# way light does at the edge of a thick lens: each spot there shows what is
# a little further out. It is done to each fresh copy of the screen behind,
# a few times a second and off the GUI thread: Windows' own blur can't take
# the shape of a small layered window (see Win32LayeredOverlayWindow).
_FROST_BLUR = 7.0
_LENS_BAND = 10.0
_LENS_SHIFT = 8.0


def frosted(
    pixels: numpy.ndarray,
    box: tuple[float, float, float, float],
    radius: float,
    screen_scale: float,
) -> numpy.ndarray:
    """``pixels`` (BGRA, as captured) blurred and bent at the rim of the body in ``box``, as RGB."""
    blurred = numpy.asarray(
        Image.fromarray(numpy.ascontiguousarray(pixels[:, :, 2::-1])).filter(
            ImageFilter.GaussianBlur(_FROST_BLUR * screen_scale),
        ),
    )
    height, width = blurred.shape[:2]
    rows, columns = numpy.mgrid[0:height, 0:width].astype(numpy.float32)
    x, y = (columns + 0.5) / screen_scale, (rows + 0.5) / screen_scale
    left, top, right, bottom = box
    across, down = x - (left + right) / 2, y - (top + bottom) / 2
    past_x = numpy.abs(across) - ((right - left) / 2 - radius)
    past_y = numpy.abs(down) - ((bottom - top) / 2 - radius)
    corner_x, corner_y = numpy.maximum(past_x, 0.0), numpy.maximum(past_y, 0.0)
    corner = numpy.hypot(corner_x, corner_y)
    # Signed distance to the rim, negative inside, and the way out through it.
    distance = corner + numpy.minimum(numpy.maximum(past_x, past_y), 0.0) - radius
    rounded = corner > 1e-6
    out_x = numpy.where(rounded, corner_x / numpy.maximum(corner, 1e-6), past_x >= past_y) * numpy.sign(across)
    out_y = numpy.where(rounded, corner_y / numpy.maximum(corner, 1e-6), past_y > past_x) * numpy.sign(down)
    nearness = numpy.where(distance < 0, numpy.clip(1.0 + distance / _LENS_BAND, 0.0, 1.0), 0.0)
    shift = _LENS_SHIFT * screen_scale * nearness**2
    source_x = numpy.clip(numpy.rint(columns + out_x * shift), 0, width - 1).astype(numpy.intp)
    source_y = numpy.clip(numpy.rint(rows + out_y * shift), 0, height - 1).astype(numpy.intp)
    return blurred[source_y, source_x]


def glazed(pixels: numpy.ndarray, behind: numpy.ndarray, coverage: numpy.ndarray) -> numpy.ndarray:
    """``pixels`` (premultiplied BGRA) over ``behind`` (RGB) wherever ``coverage`` puts the glass.

    A ``behind`` of another size, from before the window changed size, is left out.
    """
    if behind.shape[:2] != pixels.shape[:2] or coverage.shape != pixels.shape[:2]:
        return pixels
    under = coverage * (1.0 - pixels[:, :, 3].astype(numpy.float32) / 255.0)
    out = numpy.empty_like(pixels)
    out[:, :, :3] = numpy.minimum(pixels[:, :, :3] + behind[:, :, ::-1] * under[:, :, None] + 0.5, 255.0)
    out[:, :, 3] = numpy.minimum(pixels[:, :, 3] + 255.0 * under + 0.5, 255.0)
    return out


@lru_cache(maxsize=16)
def _coverage(
    size: tuple[int, int], box: tuple[int, int, int, int], radius: int, screen_size: tuple[int, int],
) -> numpy.ndarray:
    """How much of each screen pixel the body covers, 0 to 1, edged the way it is drawn."""
    covered = numpy.asarray(
        _shape(size, box, radius).resize(screen_size, Image.Resampling.LANCZOS), dtype=numpy.float32,
    ) / 255.0
    covered.flags.writeable = False
    return covered


def frost_pill(pixels: numpy.ndarray, screen_scale: float, skin: PillSkin) -> numpy.ndarray:
    """What is behind the pill's canvas, frosted for the pill at rest."""
    width, height = _RECORDING_SIZE
    return frosted(pixels, pill_box(width, height), height / 2 * skin.roundness, screen_scale)


def glaze_pill(
    pixels: numpy.ndarray, behind: numpy.ndarray, frame: Frame, screen_scale: float, skin: PillSkin,
) -> numpy.ndarray:
    """The rendered ``frame`` (premultiplied BGRA) over the frosted ``behind``, fading with the pill."""
    scale = _SUPERSAMPLE
    covered = _coverage(
        (CANVAS_WIDTH * scale, CANVAS_HEIGHT * scale),
        _snapped(tuple(edge * scale for edge in pill_box(frame.width, frame.height))),
        round(frame.height * scale / 2 * skin.roundness),
        scaled_size(CANVAS_WIDTH, CANVAS_HEIGHT, screen_scale),
    )
    return glazed(pixels, behind, covered * frame.opacity)


def _blank(image: Image.Image) -> Image.Image:
    return Image.new("RGBA", image.size, (0, 0, 0, 0))


def _fade(layer: Image.Image, strength: float) -> None:
    if strength < 1:
        layer.putalpha(layer.getchannel("A").point(lambda value: round(value * strength)))


def _blend(base: Image.Image, top: Image.Image, strength: float) -> None:
    _fade(top, strength)
    base.alpha_composite(top)


def _draw_label(draw: ImageDraw.ImageDraw, box: tuple[float, ...], skin: PillSkin) -> None:
    left, top, right, bottom = box
    draw.text(
        ((left + right) / 2, (top + bottom) / 2),
        _LABEL_TEXT,
        font=_font(_LABEL_SIZE * _SUPERSAMPLE),
        fill=(*skin.label, 255),
        anchor="mm",
    )


def _draw_controls(draw: ImageDraw.ImageDraw, frame: Frame, skin: PillSkin) -> None:
    scale = _SUPERSAMPLE
    centres = button_centres(frame)
    for button, (cx, cy, radius) in centres.items():
        hovered = button is frame.hovered
        if button is Button.FINISH:
            fill = skin.finish_hover if hovered else skin.finish
        else:
            fill = skin.button_hover if hovered else skin.button
        draw.ellipse(
            ((cx - radius) * scale, (cy - radius) * scale,
             (cx + radius) * scale, (cy + radius) * scale),
            fill=(*fill, 255),
        )
        _draw_glyph(draw, button, cx, cy, radius, frame.paused, skin)
    _draw_middle(draw, frame, centres, skin)


def _draw_glyph(
    draw: ImageDraw.ImageDraw,
    button: Button,
    cx: float,
    cy: float,
    radius: float,
    paused: bool,
    skin: PillSkin,
) -> None:
    scale = _SUPERSAMPLE
    size = radius * (0.5 if button is Button.FINISH else 0.42)
    glyph = skin.finish_glyph if button is Button.FINISH else skin.glyph
    colour = (*glyph, 255)
    width = max(round(2.2 * scale), 1)

    def point(dx: float, dy: float) -> tuple[float, float]:
        return (cx + dx) * scale, (cy + dy) * scale

    if button is Button.CANCEL:
        draw.line((point(-size, -size), point(size, size)), fill=colour, width=width)
        draw.line((point(-size, size), point(size, -size)), fill=colour, width=width)
    elif button is Button.FINISH:
        draw.line(
            (point(-size, 0.05 * size), point(-0.3 * size, 0.7 * size), point(size, -0.65 * size)),
            fill=colour, width=width, joint="curve",
        )
    elif paused:
        draw.polygon(
            (point(-0.55 * size, -size), point(-0.55 * size, size), point(size, 0)),
            fill=colour,
        )
    else:
        for dx in (-0.45 * size, 0.45 * size):
            draw.rounded_rectangle(
                (*point(dx - 0.18 * size, -0.85 * size), *point(dx + 0.18 * size, 0.85 * size)),
                radius=0.18 * size * scale,
                fill=colour,
            )


def _draw_middle(
    draw: ImageDraw.ImageDraw,
    frame: Frame,
    centres: dict[Button, tuple[float, float, float]],
    skin: PillSkin,
) -> None:
    cancel_x, middle, radius = centres[Button.CANCEL]
    pause_x = centres[Button.PAUSE][0]
    if frame.silent_microphone:
        _draw_silence_notice(
            draw,
            frame.silent_microphone,
            cancel_x + radius + _TEXT_MARGIN,
            pause_x - radius - _TEXT_MARGIN,
            middle,
            skin.warning,
        )
        return
    start = cancel_x + radius + _WAVE_MARGIN
    end = pause_x - radius - _WAVE_MARGIN
    count = _BAR_COUNT
    if frame.countdown:
        start += _draw_text(draw, frame.countdown, _COUNTDOWN_SIZE, start, middle, "lm", skin.warning)
        start += _COUNTDOWN_GAP
        count = _COUNTDOWN_BAR_COUNT
    _draw_waveform(draw, frame, start, end, middle, count, skin.wave)


def _draw_silence_notice(
    draw: ImageDraw.ImageDraw, microphone: str, start: float, end: float, middle: float, colour: Rgb,
) -> None:
    width = max(int(end - start), 0)
    centre = (start + end) / 2
    for text, y in (
        (_NOTICE_TEXT, middle - _NOTICE_LINE_OFFSET),
        (microphone, middle + _NOTICE_LINE_OFFSET),
    ):
        _draw_text(draw, _fitted(text, _NOTICE_SIZE, width), _NOTICE_SIZE, centre, y, "mm", colour)


def _draw_text(
    draw: ImageDraw.ImageDraw, text: str, size: int, x: float, y: float, anchor: str, colour: Rgb,
) -> float:
    """Draw ``text`` in ``colour`` and return how wide it is, in canvas pixels."""
    font = _font(size * _SUPERSAMPLE)
    draw.text(
        (x * _SUPERSAMPLE, y * _SUPERSAMPLE), text, font=font,
        fill=(*colour, 255), anchor=anchor,
    )
    return font.getlength(text) / _SUPERSAMPLE


@lru_cache(maxsize=256)
def _fitted(text: str, size: int, width: int) -> str:
    """``text`` cut short with an ellipsis until it fits ``width`` pixels."""
    font = _font(size * _SUPERSAMPLE)
    limit = width * _SUPERSAMPLE
    if font.getlength(text) <= limit:
        return text
    while text and font.getlength(text + "…") > limit:
        text = text[:-1]
    return text.rstrip() + "…"


def _draw_waveform(
    draw: ImageDraw.ImageDraw,
    frame: Frame,
    start: float,
    end: float,
    middle: float,
    count: int,
    colour: Rgb,
) -> None:
    """The newest slice enters past the right end and slides into the last
    slot as it fills; older ones keep sliding left and fade out past the
    first, so the row moves smoothly instead of stepping a slot at a time."""
    scale = _SUPERSAMPLE
    slot = _BAR_WIDTH + _BAR_GAP
    span = count * _BAR_WIDTH + (count - 1) * _BAR_GAP
    if end - start < span:
        return
    first = (start + end - span) / 2
    last = first + (count - 1) * slot
    tallest = max(frame.height - 2 * (_PADDING + 6), _BAR_MIN_HEIGHT)
    strength = 115 if frame.paused else 255
    levels = frame.levels[-(count + 2):]
    for back, level in enumerate(reversed(levels)):
        left = last - (back - 1 + frame.scroll) * slot
        if left > last:
            fade = 1.0 - (left - last) / slot
        elif left < first:
            fade = 1.0 - (first - left) / slot
        else:
            fade = 1.0
        if fade <= 0:
            continue
        height = _BAR_MIN_HEIGHT + (tallest - _BAR_MIN_HEIGHT) * (0.0 if frame.paused else level)
        top = middle - height / 2
        draw.rounded_rectangle(
            (left * scale, top * scale, (left + _BAR_WIDTH) * scale, (top + height) * scale),
            radius=_BAR_WIDTH * scale / 2,
            fill=(*colour, round(strength * fade)),
        )


# The notice above the pill is drawn the way the pill is: the same body,
# border, type and buttons, a first button coloured like the finish
# button, and a dot in the warning colour when something went wrong.
NOTICE_CLOSE = -1
_NOTICE_HEIGHT = _RECORDING_SIZE[1]
_NOTICE_LINE_HEIGHT = 16.0
_NOTICE_MAX_TEXT_WIDTH = 360
_NOTICE_TEXT_INSET = 18.0
_NOTICE_DOT = 7.0
_NOTICE_DOT_GAP = 8.0
_NOTICE_TEXT_GAP = 14.0
_NOTICE_BUTTON_HEIGHT = _NOTICE_HEIGHT - 2 * _PADDING
_NOTICE_BUTTON_PADDING = 14.0
# A transparent pixel all round, so the border is never cut off; glass
# leaves room for its shadow.
_NOTICE_MARGIN = 1
_GLASS_NOTICE_MARGIN = 6


@dataclass(frozen=True)
class NoticeLayout:
    """Where everything in a notice goes, in window pixels."""

    width: int
    height: int
    lines: tuple[str, ...]
    labels: tuple[str, ...]
    warning: bool
    text_left: float
    buttons: tuple[tuple[float, float, float, float], ...]
    # The close cross's centre and radius.
    close: tuple[float, float, float]
    margin: int = _NOTICE_MARGIN


def notice_layout(message: str, labels: Sequence[str], warning: bool, glass: bool = False) -> NoticeLayout:
    """One line is exactly as tall as the pill; a long message wraps instead of widening."""
    margin = _GLASS_NOTICE_MARGIN if glass else _NOTICE_MARGIN
    lines = _wrapped(message, _LABEL_SIZE, _NOTICE_MAX_TEXT_WIDTH)
    height = math.ceil(_NOTICE_HEIGHT + (len(lines) - 1) * _NOTICE_LINE_HEIGHT)
    middle = margin + height / 2
    radius = _NOTICE_BUTTON_HEIGHT / 2
    x = margin + _NOTICE_TEXT_INSET
    if warning:
        x += _NOTICE_DOT + _NOTICE_DOT_GAP
    text_left = x
    x += max(_text_width(line, _LABEL_SIZE) for line in lines) + _NOTICE_TEXT_GAP
    buttons = []
    for label in labels:
        width = _text_width(label, _LABEL_SIZE) + 2 * _NOTICE_BUTTON_PADDING
        buttons.append((x, middle - radius, x + width, middle + radius))
        x += width + _BUTTON_GAP
    close = (x + radius, middle, radius)
    x += 2 * radius + _PADDING
    return NoticeLayout(
        width=math.ceil(x) + margin,
        height=height + 2 * margin,
        lines=lines,
        labels=tuple(labels),
        warning=warning,
        text_left=text_left,
        buttons=tuple(buttons),
        close=close,
        margin=margin,
    )


def notice_target(layout: NoticeLayout, x: float, y: float) -> int | None:
    """The index of the button at ``x, y``, ``NOTICE_CLOSE`` for the cross, else None."""
    for index, (left, top, right, bottom) in enumerate(layout.buttons):
        if left <= x <= right and top <= y <= bottom:
            return index
    cx, cy, radius = layout.close
    # The same slack as the pill's buttons.
    if math.hypot(x - cx, y - cy) <= radius + 2:
        return NOTICE_CLOSE
    return None


def render_notice(
    layout: NoticeLayout,
    hovered: int | None = None,
    screen_scale: float = 1.0,
    skin: PillSkin = DEFAULT_SKIN,
) -> Image.Image:
    scale = _SUPERSAMPLE
    image = _body(
        (layout.width * scale, layout.height * scale), _notice_box(layout), _NOTICE_HEIGHT * scale / 2 * skin.roundness, skin,
    )
    draw = ImageDraw.Draw(image)
    middle = layout.height / 2
    font = _font(_LABEL_SIZE * scale)
    if layout.warning:
        dot_left = layout.text_left - _NOTICE_DOT_GAP - _NOTICE_DOT
        draw.ellipse(
            (dot_left * scale, (middle - _NOTICE_DOT / 2) * scale,
             (dot_left + _NOTICE_DOT) * scale, (middle + _NOTICE_DOT / 2) * scale),
            fill=(*skin.warning, 255),
        )
    first_line = middle - (len(layout.lines) - 1) * _NOTICE_LINE_HEIGHT / 2
    for index, line in enumerate(layout.lines):
        draw.text(
            (layout.text_left * scale, (first_line + index * _NOTICE_LINE_HEIGHT) * scale),
            line, font=font, fill=(*skin.label, 255), anchor="lm",
        )
    for index, (box, label) in enumerate(zip(layout.buttons, layout.labels, strict=True)):
        lit = index == hovered
        if index == 0:
            fill, text = (skin.finish_hover if lit else skin.finish), skin.finish_glyph
        else:
            fill, text = (skin.button_hover if lit else skin.button), skin.label
        left, top, right, bottom = (edge * scale for edge in box)
        draw.rounded_rectangle(
            (left, top, right, bottom), radius=(bottom - top) / 2 * skin.roundness, fill=(*fill, 255),
        )
        draw.text(
            ((left + right) / 2, (top + bottom) / 2), label,
            font=font, fill=(*text, 255), anchor="mm",
        )
    cx, cy, radius = layout.close
    fill = skin.button_hover if hovered == NOTICE_CLOSE else skin.button
    draw.ellipse(
        ((cx - radius) * scale, (cy - radius) * scale,
         (cx + radius) * scale, (cy + radius) * scale),
        fill=(*fill, 255),
    )
    _draw_glyph(draw, Button.CANCEL, cx, cy, radius, False, skin)
    return image.resize(scaled_size(layout.width, layout.height, screen_scale), Image.Resampling.LANCZOS)


def _notice_box(layout: NoticeLayout) -> tuple[int, int, int, int]:
    """The notice's body in supersampled pixels."""
    scale, margin = _SUPERSAMPLE, layout.margin * _SUPERSAMPLE
    return margin, margin, layout.width * scale - margin, layout.height * scale - margin


def frost_notice(pixels: numpy.ndarray, layout: NoticeLayout, screen_scale: float, skin: PillSkin) -> numpy.ndarray:
    """What is behind a notice, frosted for it."""
    left, top, right, bottom = (edge / _SUPERSAMPLE for edge in _notice_box(layout))
    return frosted(pixels, (left, top, right, bottom), _NOTICE_HEIGHT / 2 * skin.roundness, screen_scale)


def glaze_notice(
    pixels: numpy.ndarray, behind: numpy.ndarray, layout: NoticeLayout, screen_scale: float, skin: PillSkin,
) -> numpy.ndarray:
    """A rendered notice (premultiplied BGRA) over the frosted ``behind``."""
    scale = _SUPERSAMPLE
    covered = _coverage(
        (layout.width * scale, layout.height * scale),
        _notice_box(layout),
        round(_NOTICE_HEIGHT * scale / 2 * skin.roundness),
        scaled_size(layout.width, layout.height, screen_scale),
    )
    return glazed(pixels, behind, covered)


def _wrapped(text: str, size: int, width: int) -> tuple[str, ...]:
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}" if line else word
        if line and _text_width(candidate, size) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    # A single word too long for a line is cut short rather than overflowing.
    # No character is under a pixel wide, so a line never needs more
    # characters than pixels, and the cut never searches far.
    return tuple(_fitted(line[:width], size, width) for line in lines)


def _text_width(text: str, size: int) -> float:
    return _font(size * _SUPERSAMPLE).getlength(text) / _SUPERSAMPLE


@cache
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Segoe UI Semibold ships as seguisb.ttf.
    for name in ("seguisb.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)
