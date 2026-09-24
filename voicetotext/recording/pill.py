"""Shape, motion and drawing of the recording pill. No Win32 and no Tk here.

The pill lives on a fixed transparent canvas and only its drawn shape
changes size, so the window never moves or resizes mid-animation and the
transparent margin passes clicks through to whatever is underneath.
"""

import math
from dataclasses import dataclass
from enum import Enum
from functools import cache

import numpy
from PIL import Image, ImageDraw, ImageFont

CANVAS_WIDTH = 248
CANVAS_HEIGHT = 72
# The pill's bottom edge stays put and it grows upward, so the canvas keeps
# a little room below it for the overshoot of the spring.
_BOTTOM_INSET = 10
_SUPERSAMPLE = 3

_RECORDING_SIZE = (220.0, 48.0)
_TRANSCRIBING_SIZE = (156.0, 34.0)
_COLLAPSED_SIZE = (44.0, 10.0)

_PADDING = 8.0
_BUTTON_GAP = 6.0
_WAVE_MARGIN = 12.0
_BAR_COUNT = 11
_BAR_WIDTH = 3.0
_BAR_GAP = 4.0
_BAR_MIN_HEIGHT = 3.0
_BREATHING_RATE = 0.18

_RECORDING_FILL = (12, 12, 14)
_TRANSCRIBING_FILL = (38, 38, 42)
_PROGRESS_FILL = (92, 92, 104)
_BORDER = (255, 255, 255, 26)
_BUTTON_FILL = (58, 58, 62)
_BUTTON_HOVER_FILL = (82, 82, 88)
_FINISH_FILL = (255, 255, 255)
_FINISH_HOVER_FILL = (222, 222, 226)
_GLYPH = (255, 255, 255)
_FINISH_GLYPH = (10, 10, 12)
_LABEL = (236, 236, 240)
_LABEL_TEXT = "Transcribing"
_LABEL_SIZE = 12

# A little under critical damping, so the pill lands with one small
# bounce the way Typeless's does, then settles.
_STIFFNESS = 380.0
_DAMPING = 26.0
_SPRING_STEP_SECONDS = 1 / 240


class Mode(Enum):
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    CLOSED = "closed"


class Button(Enum):
    CANCEL = "cancel"
    PAUSE = "pause"
    FINISH = "finish"


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
    level: float = 0.0
    phase: float = 0.0
    paused: bool = False
    progress: float = 0.0
    hovered: Button | None = None


def target_size(mode: Mode) -> tuple[float, float]:
    if mode is Mode.RECORDING:
        return _RECORDING_SIZE
    if mode is Mode.TRANSCRIBING:
        return _TRANSCRIBING_SIZE
    return _COLLAPSED_SIZE


def collapsed_size() -> tuple[float, float]:
    return _COLLAPSED_SIZE


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


def transcription_progress(elapsed: float, estimate: float | None) -> float:
    """How full the bar should be ``elapsed`` seconds into a transcription.

    With an estimate the fill is honest: linear against the estimate up to
    95%, then creeping, so a slow run never looks finished. Without one
    there is nothing to be honest about, so it eases towards 90%.
    """
    elapsed = max(elapsed, 0.0)
    if estimate is None or estimate <= 0:
        return 0.9 * (1 - math.exp(-elapsed / 0.8))
    if elapsed <= estimate:
        return 0.95 * elapsed / estimate
    return 0.95 + 0.04 * (1 - math.exp(-(elapsed - estimate) / estimate))


def render(frame: Frame) -> Image.Image:
    scale = _SUPERSAMPLE
    image = Image.new(
        "RGBA", (CANVAS_WIDTH * scale, CANVAS_HEIGHT * scale), (0, 0, 0, 0),
    )
    if frame.opacity > 0.004 and frame.width > 1 and frame.height > 1:
        _draw_pill(image, frame)
    return image.resize((CANVAS_WIDTH, CANVAS_HEIGHT), Image.Resampling.LANCZOS)


def _draw_pill(image: Image.Image, frame: Frame) -> None:
    scale = _SUPERSAMPLE
    box = tuple(edge * scale for edge in pill_box(frame.width, frame.height))
    radius = frame.height * scale / 2
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fill = _mix(_RECORDING_FILL, _TRANSCRIBING_FILL, frame.label)
    draw.rounded_rectangle(box, radius=radius, fill=(*fill, 250), outline=_BORDER, width=scale)
    # ImageDraw replaces pixels rather than blending them, so anything
    # drawn part-transparent would punch a hole through the pill. Each
    # fading part is drawn opaque on its own layer and blended in instead.
    if frame.label > 0:
        _draw_progress(layer, box, radius, frame)
        label = _blank(image)
        _draw_label(ImageDraw.Draw(label), box)
        # The label waits for the buttons to be mostly gone, so the two
        # never sit on top of each other mid-change.
        _blend(layer, label, frame.label * max(0.0, 1.0 - 2.0 * frame.controls))
    if frame.controls > 0:
        controls = _blank(image)
        _draw_controls(ImageDraw.Draw(controls), frame)
        _blend(layer, controls, frame.controls)
    _fade(layer, frame.opacity)
    image.alpha_composite(layer)


def _blank(image: Image.Image) -> Image.Image:
    return Image.new("RGBA", image.size, (0, 0, 0, 0))


def _fade(layer: Image.Image, strength: float) -> None:
    if strength < 1:
        layer.putalpha(layer.getchannel("A").point(lambda value: round(value * strength)))


def _blend(base: Image.Image, top: Image.Image, strength: float) -> None:
    _fade(top, strength)
    base.alpha_composite(top)


def _draw_progress(
    layer: Image.Image,
    box: tuple[float, ...],
    radius: float,
    frame: Frame,
) -> None:
    left, top, right, bottom = box
    filled = left + (right - left) * min(max(frame.progress, 0.0), 1.0)
    if filled <= left:
        return
    shape = Image.new("L", layer.size, 0)
    ImageDraw.Draw(shape).rounded_rectangle(box, radius=radius, fill=255)
    bar = Image.new("L", layer.size, 0)
    ImageDraw.Draw(bar).rectangle((left, top, filled, bottom), fill=round(255 * frame.label))
    mask = Image.fromarray(numpy.minimum(numpy.asarray(shape), numpy.asarray(bar)))
    tint = Image.new("RGBA", layer.size, (*_PROGRESS_FILL, 255))
    tint.putalpha(mask)
    layer.alpha_composite(tint)


def _draw_label(draw: ImageDraw.ImageDraw, box: tuple[float, ...]) -> None:
    left, top, right, bottom = box
    draw.text(
        ((left + right) / 2, (top + bottom) / 2),
        _LABEL_TEXT,
        font=_font(_LABEL_SIZE * _SUPERSAMPLE),
        fill=(*_LABEL, 255),
        anchor="mm",
    )


def _draw_controls(draw: ImageDraw.ImageDraw, frame: Frame) -> None:
    scale = _SUPERSAMPLE
    centres = button_centres(frame)
    for button, (cx, cy, radius) in centres.items():
        hovered = button is frame.hovered
        if button is Button.FINISH:
            fill = _FINISH_HOVER_FILL if hovered else _FINISH_FILL
        else:
            fill = _BUTTON_HOVER_FILL if hovered else _BUTTON_FILL
        draw.ellipse(
            ((cx - radius) * scale, (cy - radius) * scale,
             (cx + radius) * scale, (cy + radius) * scale),
            fill=(*fill, 255),
        )
        _draw_glyph(draw, button, cx, cy, radius, frame.paused)
    _draw_waveform(draw, frame, centres)


def _draw_glyph(
    draw: ImageDraw.ImageDraw,
    button: Button,
    cx: float,
    cy: float,
    radius: float,
    paused: bool,
) -> None:
    scale = _SUPERSAMPLE
    size = radius * (0.5 if button is Button.FINISH else 0.42)
    glyph = _FINISH_GLYPH if button is Button.FINISH else _GLYPH
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


def _draw_waveform(
    draw: ImageDraw.ImageDraw,
    frame: Frame,
    centres: dict[Button, tuple[float, float, float]],
) -> None:
    scale = _SUPERSAMPLE
    cancel_x, middle, radius = centres[Button.CANCEL]
    pause_x = centres[Button.PAUSE][0]
    start = cancel_x + radius + _WAVE_MARGIN
    end = pause_x - radius - _WAVE_MARGIN
    span = _BAR_COUNT * _BAR_WIDTH + (_BAR_COUNT - 1) * _BAR_GAP
    if end - start < span:
        return
    first = (start + end - span) / 2
    tallest = max(frame.height - 2 * (_PADDING + 6), _BAR_MIN_HEIGHT)
    alpha = 115 if frame.paused else 255
    for index in range(_BAR_COUNT):
        height = (
            _BAR_MIN_HEIGHT if frame.paused
            else _bar_height(index, frame.level, frame.phase, tallest)
        )
        left = first + index * (_BAR_WIDTH + _BAR_GAP)
        top = middle - height / 2
        draw.rounded_rectangle(
            (left * scale, top * scale, (left + _BAR_WIDTH) * scale, (top + height) * scale),
            radius=_BAR_WIDTH * scale / 2,
            fill=(*_GLYPH, alpha),
        )


def _bar_height(index: int, level: float, phase: float, tallest: float) -> float:
    half = (_BAR_COUNT - 1) / 2
    distance = abs(index - half)
    bias = 1.0 - (0.62 * ((distance / half) ** 1.35))
    pulse = 0.55 + (0.45 * math.sin(phase + (distance * 0.8)))
    breathing = 0.06 + (0.06 * math.sin(phase * _BREATHING_RATE))
    amplitude = max(level * pulse, breathing) * bias
    return _BAR_MIN_HEIGHT + ((tallest - _BAR_MIN_HEIGHT) * amplitude)


def _mix(
    first: tuple[int, int, int], second: tuple[int, int, int], amount: float,
) -> tuple[int, int, int]:
    amount = min(max(amount, 0.0), 1.0)
    return tuple(
        round(a + (b - a) * amount) for a, b in zip(first, second, strict=True)
    )


@cache
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("segoeuisb.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)
