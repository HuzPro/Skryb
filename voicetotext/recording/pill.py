"""Shape, motion and drawing of the recording pill. No Win32 and no Tk here.

The pill lives on a fixed transparent canvas and only its drawn shape
changes size, so the window never moves or resizes mid-animation and the
transparent margin passes clicks through to whatever is underneath.
"""

import math
from dataclasses import dataclass
from enum import Enum
from functools import cache, lru_cache

import numpy
from PIL import Image, ImageDraw, ImageFont

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
# The dark theme accent, blended into the bar so the white label stays readable.
_PROGRESS_FILL = (76, 96, 138)
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
# The caption sits on the pill's top edge like a badge: half above it, half
# over the empty padding, so it never covers a button or the waveform.
_CAPTION_SIZE = 9
_CAPTION_HEIGHT = 12.0
_CAPTION_PADDING = 7.0
# The dark theme warning colour (#fbbf24).
_AMBER = (251, 191, 36)
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


class Mode(Enum):
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    CLOSED = "closed"


class Button(Enum):
    CANCEL = "cancel"
    PAUSE = "pause"
    FINISH = "finish"


# The waveform scales to how loud you have been lately rather than to a
# fixed gain, so speech from across the room fills the bars the way
# speech up close does. Below the gate is room noise and draws flat.
_NOISE_GATE = 0.003
_QUIETEST_LOUDNESS = 0.008
_LOUDNESS_HALF_LIFE_SECONDS = 4.0
_STARTING_LOUDNESS = 0.06


class VoiceLevel:
    """Microphone loudness mapped to 0..1 against the speaker's own recent peak.

    The peak rises at once to anything louder and eases back down over a
    few seconds, so stepping away from the microphone rescales the bars
    within a sentence while a pause between words does not.
    """

    def __init__(self) -> None:
        self._loudness = _STARTING_LOUDNESS

    def update(self, rms: float, seconds: float) -> float:
        if not math.isfinite(rms) or rms < 0:
            rms = 0.0
        decay = 0.5 ** (max(seconds, 0.0) / _LOUDNESS_HALF_LIFE_SECONDS)
        self._loudness = max(rms, self._loudness * decay, _QUIETEST_LOUDNESS)
        return min(max((rms - _NOISE_GATE) / (self._loudness - _NOISE_GATE), 0.0), 1.0)


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
    caption: str = ""
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


# With an estimate the bar moves at a steady, honest pace until three
# quarters full, then eases towards 99% along a curve that starts at the
# same speed, so there is no visible kink where one hands over to the
# other and a slow run never looks finished.
_STEADY_PACE = 0.9
_STEADY_UNTIL = 0.75
_EASE_TOWARDS = 0.99
# Without an estimate there is nothing to be honest about.
_UNKNOWN_EASE_TOWARDS = 0.9
_UNKNOWN_EASE_SECONDS = 0.8


def transcription_progress(elapsed: float, estimate: float | None) -> float:
    """How full the bar should be ``elapsed`` seconds into a transcription.

    Smooth in time: both the fill and its speed are continuous, which is
    what keeps the edge from visibly stepping between frames.
    """
    elapsed = max(elapsed, 0.0)
    if estimate is None or estimate <= 0:
        return _UNKNOWN_EASE_TOWARDS * (1 - math.exp(-elapsed / _UNKNOWN_EASE_SECONDS))
    pace = _STEADY_PACE / estimate
    handover = _STEADY_UNTIL / pace
    if elapsed <= handover:
        return pace * elapsed
    remaining = _EASE_TOWARDS - _STEADY_UNTIL
    return _EASE_TOWARDS - remaining * math.exp(-(elapsed - handover) * pace / remaining)


def finishing_progress(start: float, elapsed: float, duration: float) -> float:
    """From ``start`` to full over ``duration`` seconds, slowing into the end."""
    if duration <= 0 or elapsed >= duration:
        return 1.0
    t = max(elapsed, 0.0) / duration
    return start + (1.0 - start) * (1 - (1 - t) ** 3)


class SettledBar:
    """The transcribing bar at rest, drawn once, filled by slicing.

    Redrawing the whole pill costs milliseconds; a frame at 240 Hz has
    four. Once the pill has settled into the bar only the fill moves, so
    the bar is drawn empty and full a single time and each frame joins
    the two at the fill edge, blending the one column the edge falls in
    so it moves by fractions of a pixel instead of whole-pixel steps.
    """

    def __init__(self, to_pixels) -> None:
        width, height = _TRANSCRIBING_SIZE
        blank = Frame(width, height, opacity=1.0, controls=0.0, label=1.0)
        self._empty = to_pixels(render(Frame(**{**blank.__dict__, "progress": 0.0})))
        self._full = to_pixels(render(Frame(**{**blank.__dict__, "progress": 1.0})))
        left, _top, right, _bottom = pill_box(width, height)
        self._left = left
        self._span = right - left

    def pixels(self, progress: float) -> bytes:
        if progress <= 0:
            return self._empty.tobytes()
        if progress >= 1:
            return self._full.tobytes()
        edge = self._left + self._span * progress
        column = math.floor(edge)
        share = edge - column
        out = self._empty.copy()
        out[:, :column] = self._full[:, :column]
        if 0 <= column < out.shape[1] and share > 0:
            mixed = (
                self._full[:, column].astype(numpy.uint16) * round(share * 256)
                + self._empty[:, column].astype(numpy.uint16) * (256 - round(share * 256))
            ) >> 8
            out[:, column] = mixed.astype(numpy.uint8)
        return out.tobytes()


def is_settled_bar(frame: Frame) -> bool:
    width, height = _TRANSCRIBING_SIZE
    return (
        abs(frame.width - width) < 0.05
        and abs(frame.height - height) < 0.05
        and frame.opacity > 0.999
        and frame.controls < 0.001
        and frame.label > 0.999
    )


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
        if frame.caption:
            caption = _blank(image)
            _draw_caption(ImageDraw.Draw(caption), box, frame.caption)
            _blend(layer, caption, frame.controls)
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


def _draw_caption(draw: ImageDraw.ImageDraw, box: tuple[float, ...], caption: str) -> None:
    scale = _SUPERSAMPLE
    left, top, right, _bottom = box
    font = _font(_CAPTION_SIZE * scale)
    text_left, _, text_right, _ = draw.textbbox((0, 0), caption, font=font)
    half_width = (text_right - text_left) / 2 + _CAPTION_PADDING * scale
    half_height = _CAPTION_HEIGHT * scale / 2
    centre = (left + right) / 2
    draw.rounded_rectangle(
        (centre - half_width, top - half_height, centre + half_width, top + half_height),
        radius=half_height,
        fill=(*_TRANSCRIBING_FILL, 255),
        outline=_BORDER,
        width=scale,
    )
    draw.text((centre, top), caption, font=font, fill=(*_LABEL, 255), anchor="mm")


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
    _draw_middle(draw, frame, centres)


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


def _draw_middle(
    draw: ImageDraw.ImageDraw,
    frame: Frame,
    centres: dict[Button, tuple[float, float, float]],
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
        )
        return
    start = cancel_x + radius + _WAVE_MARGIN
    end = pause_x - radius - _WAVE_MARGIN
    count = _BAR_COUNT
    if frame.countdown:
        start += _draw_text(draw, frame.countdown, _COUNTDOWN_SIZE, start, middle, "lm")
        start += _COUNTDOWN_GAP
        count = _COUNTDOWN_BAR_COUNT
    _draw_waveform(draw, frame, start, end, middle, count)


def _draw_silence_notice(
    draw: ImageDraw.ImageDraw, microphone: str, start: float, end: float, middle: float,
) -> None:
    width = max(int(end - start), 0)
    centre = (start + end) / 2
    for text, y in (
        (_NOTICE_TEXT, middle - _NOTICE_LINE_OFFSET),
        (microphone, middle + _NOTICE_LINE_OFFSET),
    ):
        _draw_text(draw, _fitted(text, _NOTICE_SIZE, width), _NOTICE_SIZE, centre, y, "mm")


def _draw_text(
    draw: ImageDraw.ImageDraw, text: str, size: int, x: float, y: float, anchor: str,
) -> float:
    """Draw ``text`` in amber and return how wide it is, in canvas pixels."""
    font = _font(size * _SUPERSAMPLE)
    draw.text(
        (x * _SUPERSAMPLE, y * _SUPERSAMPLE), text, font=font,
        fill=(*_AMBER, 255), anchor=anchor,
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
) -> None:
    scale = _SUPERSAMPLE
    span = count * _BAR_WIDTH + (count - 1) * _BAR_GAP
    if end - start < span:
        return
    first = (start + end - span) / 2
    tallest = max(frame.height - 2 * (_PADDING + 6), _BAR_MIN_HEIGHT)
    alpha = 115 if frame.paused else 255
    for index in range(count):
        height = (
            _BAR_MIN_HEIGHT if frame.paused
            else _bar_height(index, count, frame.level, frame.phase, tallest)
        )
        left = first + index * (_BAR_WIDTH + _BAR_GAP)
        top = middle - height / 2
        draw.rounded_rectangle(
            (left * scale, top * scale, (left + _BAR_WIDTH) * scale, (top + height) * scale),
            radius=_BAR_WIDTH * scale / 2,
            fill=(*_GLYPH, alpha),
        )


def _bar_height(
    index: int, count: int, level: float, phase: float, tallest: float,
) -> float:
    half = max((count - 1) / 2, 1.0)
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
