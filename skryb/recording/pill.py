"""Shape, motion and drawing of the recording pill. No Win32 and no Tk here.

The pill lives on a fixed transparent canvas and only its drawn shape
changes size, so the window never moves or resizes mid-animation and the
transparent margin passes clicks through to whatever is underneath.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache, lru_cache

import numpy
from PIL import Image, ImageDraw, ImageFont

from skryb.core.brand_mark import EMERALD_DEEP, EMERALD_LIGHT

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

_PILL_FILL = (12, 12, 14)
_BADGE_FILL = (38, 38, 42)
# The brand emerald: the voice bars, and the colour the transcribing shimmer lifts the fill towards.
_WAVE = EMERALD_LIGHT
_SHIMMER_FILL = EMERALD_DEEP
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


# While transcribing, a soft band of the accent colour sweeps across the
# pill, easing in and out of each pass. It only lifts the dark fill
# towards the band colour, never darkens anything, so the white label
# reads the same throughout. Applied to finished pixels with numpy: a
# full redraw costs milliseconds, and a frame at 240 Hz has four.
_SHIMMER_SECONDS = 0.9
_SHIMMER_WIDTH = 130.0
_SHIMMER_STRENGTH = 0.6


def shimmer_position(seconds: float) -> float:
    """Where the band is ``seconds`` into transcribing: 0 to 1, repeating."""
    t = (max(seconds, 0.0) / _SHIMMER_SECONDS) % 1.0
    return t * t * (3 - 2 * t)


def shimmered(pixels: numpy.ndarray, frame: Frame) -> numpy.ndarray:
    """``pixels`` (premultiplied BGRA) with the shimmer band drawn at ``frame.shimmer``."""
    if frame.shimmer < 0 or frame.label <= 0:
        return pixels
    left, _top, right, _bottom = pill_box(frame.width, frame.height)
    centre = left - _SHIMMER_WIDTH / 2 + (right - left + _SHIMMER_WIDTH) * frame.shimmer
    x = numpy.arange(pixels.shape[1], dtype=numpy.float32) + 0.5
    distance = numpy.clip(numpy.abs(x - centre) / (_SHIMMER_WIDTH / 2), 0.0, 1.0)
    weight = (numpy.cos(distance * math.pi) * 0.5 + 0.5) * (_SHIMMER_STRENGTH * frame.label)
    columns = numpy.nonzero(weight > 0.002)[0]
    if columns.size == 0:
        return pixels
    span = slice(int(columns[0]), int(columns[-1]) + 1)
    region = pixels[:, span, :].astype(numpy.float32)
    colour = numpy.array(_SHIMMER_FILL[::-1], dtype=numpy.float32) * (region[:, :, 3:4] / 255.0)
    rgb = region[:, :, :3]
    lifted = rgb + (colour - rgb) * weight[span][None, :, None]
    region[:, :, :3] = numpy.maximum(rgb, lifted)
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
    draw.rounded_rectangle(
        box, radius=radius, fill=(*_PILL_FILL, 250), outline=_BORDER, width=scale,
    )
    # ImageDraw replaces pixels rather than blending them, so anything
    # drawn part-transparent would punch a hole through the pill. Each
    # fading part is drawn opaque on its own layer and blended in instead.
    if frame.label > 0:
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
        fill=(*_BADGE_FILL, 255),
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
            fill=(*_WAVE, round(strength * fade)),
        )


# The notice above the pill is drawn the way the pill is: the same dark
# capsule, border, type and round buttons, a white first button like the
# finish button, and an amber dot when something went wrong.
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
# A transparent pixel all round, so the border is never cut off.
_NOTICE_MARGIN = 1


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


def notice_layout(message: str, labels: Sequence[str], warning: bool) -> NoticeLayout:
    """One line is exactly as tall as the pill; a long message wraps instead of widening."""
    lines = _wrapped(message, _LABEL_SIZE, _NOTICE_MAX_TEXT_WIDTH)
    height = math.ceil(_NOTICE_HEIGHT + (len(lines) - 1) * _NOTICE_LINE_HEIGHT)
    middle = _NOTICE_MARGIN + height / 2
    radius = _NOTICE_BUTTON_HEIGHT / 2
    x = _NOTICE_MARGIN + _NOTICE_TEXT_INSET
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
        width=math.ceil(x) + _NOTICE_MARGIN,
        height=height + 2 * _NOTICE_MARGIN,
        lines=lines,
        labels=tuple(labels),
        warning=warning,
        text_left=text_left,
        buttons=tuple(buttons),
        close=close,
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


def render_notice(layout: NoticeLayout, hovered: int | None = None) -> Image.Image:
    scale = _SUPERSAMPLE
    image = Image.new("RGBA", (layout.width * scale, layout.height * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    margin = _NOTICE_MARGIN * scale
    draw.rounded_rectangle(
        (margin, margin, layout.width * scale - margin, layout.height * scale - margin),
        radius=_NOTICE_HEIGHT * scale / 2,
        fill=(*_PILL_FILL, 250),
        outline=_BORDER,
        width=scale,
    )
    middle = layout.height / 2
    font = _font(_LABEL_SIZE * scale)
    if layout.warning:
        dot_left = layout.text_left - _NOTICE_DOT_GAP - _NOTICE_DOT
        draw.ellipse(
            (dot_left * scale, (middle - _NOTICE_DOT / 2) * scale,
             (dot_left + _NOTICE_DOT) * scale, (middle + _NOTICE_DOT / 2) * scale),
            fill=(*_AMBER, 255),
        )
    first_line = middle - (len(layout.lines) - 1) * _NOTICE_LINE_HEIGHT / 2
    for index, line in enumerate(layout.lines):
        draw.text(
            (layout.text_left * scale, (first_line + index * _NOTICE_LINE_HEIGHT) * scale),
            line, font=font, fill=(*_LABEL, 255), anchor="lm",
        )
    for index, (box, label) in enumerate(zip(layout.buttons, layout.labels, strict=True)):
        lit = index == hovered
        if index == 0:
            fill, text = (_FINISH_HOVER_FILL if lit else _FINISH_FILL), _FINISH_GLYPH
        else:
            fill, text = (_BUTTON_HOVER_FILL if lit else _BUTTON_FILL), _LABEL
        left, top, right, bottom = (edge * scale for edge in box)
        draw.rounded_rectangle(
            (left, top, right, bottom), radius=(bottom - top) / 2, fill=(*fill, 255),
        )
        draw.text(
            ((left + right) / 2, (top + bottom) / 2), label,
            font=font, fill=(*text, 255), anchor="mm",
        )
    cx, cy, radius = layout.close
    fill = _BUTTON_HOVER_FILL if hovered == NOTICE_CLOSE else _BUTTON_FILL
    draw.ellipse(
        ((cx - radius) * scale, (cy - radius) * scale,
         (cx + radius) * scale, (cy + radius) * scale),
        fill=(*fill, 255),
    )
    _draw_glyph(draw, Button.CANCEL, cx, cy, radius, paused=False)
    return image.resize((layout.width, layout.height), Image.Resampling.LANCZOS)


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
