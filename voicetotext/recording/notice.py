"""A small notice where the pill was, drawn the way the pill is, that never takes the focus."""

import time
import tkinter
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy

from voicetotext.recording.overlay import (
    MouseHandler,
    Win32LayeredOverlayWindow,
    premultiplied_array,
)
from voicetotext.recording.pill import (
    CANVAS_WIDTH,
    DEFAULT_PILL_POSITION,
    NOTICE_CLOSE,
    Mode,
    NoticeLayout,
    PillPosition,
    Spring,
    notice_layout,
    notice_target,
    pill_box,
    pill_origin,
    render_notice,
    target_size,
)

NoticeAction = tuple[str, Callable[[], None]]

_FRAME_MS = 16
_SHOW_SECONDS = 10.0
# Tucked behind a pill that is busy recording, a notice that is left alone
# goes sooner; pointing at it holds it, and it waits a moment once the
# pointer leaves.
_TUCKED_SECONDS = 4.0
_AFTER_HOVER_SECONDS = 3.0
_RAISED_LINGER_SECONDS = 1.2
# How much of a tucked notice shows past the pill's edge, and how faint it is.
# Only that strip is drawn: the notice is wider than the pill, and its text
# cut off by the pill's edge read as clutter on either side of it.
_TUCKED_SHOWING = 0.25
_TUCKED_OPACITY = 0.7
_MAX_BUTTONS = 3
_GAP_FROM_PILL = 8
_GONE = 0.01


class NoticeWindowApi(Protocol):
    """The native half of the notice; tests swap in a fake."""

    def work_area_for_foreground(self) -> tuple[int, int, int, int]: ...
    def resize(self, width: int, height: int) -> None: ...
    def show_at(self, x: int, y: int) -> None: ...
    def move_to(self, x: int, y: int) -> None: ...
    def push_frame(self, pixels: bytes) -> None: ...
    def hide(self) -> None: ...
    def dispose(self) -> None: ...
    def set_mouse_handler(self, handler: MouseHandler) -> None: ...
    def cursor_position(self) -> tuple[int, int] | None: ...
    def set_hand_cursor(self, hand: bool) -> None: ...


def _default_position() -> PillPosition:
    return DEFAULT_PILL_POSITION


def _no_pill() -> bool:
    return False


class Notice:
    """One message with up to three buttons and a close cross, where the pill was.

    GUI thread only; other threads hop over with the GUI dispatcher. It is
    a layered popup like the pill's own window, so it never takes the
    focus from the app the text is going to, and only its drawn shape
    takes clicks. It opens centred on the pill's place, wherever the pill
    was dragged. While the pill is up, recording or transcribing, it slides
    up and tucks behind it, half showing and a little faint; pointing at
    it brings it out in full above the pill. It only ever moves without
    changing its stacking, so the pill, shown after it, stays in front. A
    new notice replaces the one showing. Each fades away on its own: after
    ten seconds, or four once tucked, held for as long as it is pointed at.
    """

    def __init__(
        self,
        root: tkinter.Misc,
        window_factory: Callable[[], NoticeWindowApi] = Win32LayeredOverlayWindow,
        *,
        load_position: Callable[[], PillPosition] = _default_position,
        pill_showing: Callable[[], bool] = _no_pill,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = root
        self._window_factory = window_factory
        self._window: NoticeWindowApi | None = None
        self._load_position = load_position
        self._pill_showing = pill_showing
        self._clock = clock
        self._layout: NoticeLayout | None = None
        self._actions: list[Callable[[], None]] = []
        self._hovered: int | None = None
        self._pressed: int | None = None
        self._clicked: int | None = None
        self._tick_after: str | None = None
        self._message = ""
        self._showing = False
        self._fading = False
        self._last_tick = 0.0
        self._fade_at = 0.0
        self._raised_until = 0.0
        self._was_tucked = False
        self._behind = False
        self._work_area = (0, 0, 0, 0)
        self._x = 0
        self._y = Spring(0.0, damping=40.0)
        self._opacity = Spring(0.0, damping=40.0)
        self._pixels: dict[int | None, numpy.ndarray] = {}
        self._pushed: tuple[int | None, int, tuple[int, int] | None] | None = None

    @property
    def visible(self) -> bool:
        return self._showing

    @property
    def message(self) -> str:
        return self._message

    @property
    def layout(self) -> NoticeLayout | None:
        return self._layout

    @property
    def opacity(self) -> float:
        return self._opacity.value if self._showing else 0.0

    def show(
        self, message: str, buttons: Sequence[NoticeAction] = (), *, warning: bool = False,
    ) -> None:
        self.hide()
        actions = list(buttons)[:_MAX_BUTTONS]
        layout = notice_layout(message, [label for label, _action in actions], warning)
        self._layout = layout
        self._actions = [action for _label, action in actions]
        self._message = message
        self._hovered = None
        self._pressed = None
        self._clicked = None
        self._pixels = {}
        self._pushed = None
        window = self._api()
        window.resize(layout.width, layout.height)
        self._work_area = window.work_area_for_foreground()
        tucked = self._pill_showing()
        self._x, y = self._place(layout, tucked=tucked, raised=False)
        self._y.value = self._y.target = float(y)
        self._opacity.value, self._opacity.target = 0.0, self._resting_opacity(tucked)
        now = self._clock()
        self._last_tick = now
        self._was_tucked = tucked
        self._behind = tucked
        self._fade_at = now + (_TUCKED_SECONDS if tucked else _SHOW_SECONDS)
        self._raised_until = 0.0
        self._fading = False
        self._showing = True
        window.show_at(self._x, y)
        self._draw()
        self._tick_after = self._root.after(_FRAME_MS, self._tick)

    def hide(self) -> None:
        """Put the notice away at once."""
        if not self._showing:
            return
        self._showing = False
        if self._tick_after is not None:
            self._root.after_cancel(self._tick_after)
            self._tick_after = None
        self._api().hide()

    def dispose(self) -> None:
        self.hide()
        if self._window is not None:
            self._window.dispose()
            self._window = None

    def _api(self) -> NoticeWindowApi:
        if self._window is None:
            self._window = self._window_factory()
            self._window.set_mouse_handler(self._on_mouse)
        return self._window

    def _place(self, layout: NoticeLayout, *, tucked: bool, raised: bool) -> tuple[int, int]:
        left, top, right, _bottom = self._work_area
        pill_x, pill_y = pill_origin(self._load_position(), self._work_area)
        _box_left, box_top, _box_right, box_bottom = pill_box(*target_size(Mode.RECORDING))
        x = pill_x + (CANVAS_WIDTH - layout.width) // 2
        x = min(max(x, left), max(right - layout.width, left))
        pill_top, pill_bottom = pill_y + round(box_top), pill_y + round(box_bottom)
        if not tucked:
            return x, round((pill_top + pill_bottom - layout.height) / 2)
        above = pill_top - _GAP_FROM_PILL - layout.height >= top
        if raised:
            return x, (
                pill_top - _GAP_FROM_PILL - layout.height if above
                else pill_bottom + _GAP_FROM_PILL
            )
        showing = round(layout.height * _TUCKED_SHOWING)
        return x, pill_top - showing if above else pill_bottom + showing - layout.height

    def _resting_opacity(self, tucked: bool) -> float:
        return _TUCKED_OPACITY if tucked else 1.0

    def _tick(self) -> None:
        self._tick_after = None
        if not self._showing or self._layout is None:
            return
        now = self._clock()
        seconds = min(max(now - self._last_tick, 0.0), 0.05)
        self._last_tick = now
        if self._act_on_click():
            return
        self._follow(now)
        self._y.step(seconds)
        self._opacity.step(seconds)
        if self._fading and self._opacity.value < _GONE:
            self.hide()
            return
        self._api().move_to(self._x, round(self._y.value))
        self._draw()
        self._tick_after = self._root.after(_FRAME_MS, self._tick)

    def _act_on_click(self) -> bool:
        clicked, self._clicked = self._clicked, None
        if clicked is None:
            return False
        self.hide()
        if clicked != NOTICE_CLOSE:
            self._actions[clicked]()
        return True

    def _follow(self, now: float) -> None:
        tucked = self._pill_showing()
        if tucked and not self._was_tucked:
            self._fade_at = min(self._fade_at, now + _TUCKED_SECONDS)
        self._was_tucked = tucked
        hovered, pointed_at = self._pointer(tucked)
        if hovered != self._hovered:
            self._hovered = hovered
            self._api().set_hand_cursor(hovered is not None)
        if pointed_at:
            self._raised_until = now + _RAISED_LINGER_SECONDS
            self._fade_at = max(self._fade_at, now + _AFTER_HOVER_SECONDS)
            self._fading = False
        raised = tucked and now < self._raised_until
        self._behind = tucked and not raised
        self._y.target = float(self._place(self._layout, tucked=tucked, raised=raised)[1])
        if not self._fading and now >= self._fade_at:
            self._fading = True
        self._opacity.target = (
            0.0 if self._fading else 1.0 if raised else self._resting_opacity(tucked)
        )

    def _pointer(self, tucked: bool) -> tuple[int | None, bool]:
        """The button under the pointer, and whether the pointer is on the notice at all.

        Tucked, only the part showing past the pill counts: the rest is behind it.
        """
        position = self._api().cursor_position()
        if position is None or self._layout is None:
            return None, False
        x, y = position
        if not (0 <= x < self._layout.width and 0 <= y < self._layout.height):
            return None, False
        hidden = self._hidden_rows()
        if hidden is not None and hidden[0] <= y < hidden[1]:
            return None, False
        return notice_target(self._layout, x, y), True

    def _hidden_rows(self) -> tuple[int, int] | None:
        """The rows not drawn while it rests behind the pill: all but the strip past its edge."""
        if not self._behind or self._layout is None:
            return None
        pill_y = pill_origin(self._load_position(), self._work_area)[1]
        _left, top, _right, bottom = pill_box(*target_size(Mode.RECORDING))
        y = round(self._y.value)
        if y < pill_y + top:
            return max(pill_y + round(top) - y, 0), self._layout.height
        return 0, min(max(pill_y + round(bottom) - y, 0), self._layout.height)

    def _draw(self) -> None:
        if self._layout is None:
            return
        level = round(self._opacity.value * 255)
        hidden = self._hidden_rows()
        if self._pushed == (self._hovered, level, hidden):
            return
        pixels = self._pixels.get(self._hovered)
        if pixels is None:
            pixels = premultiplied_array(render_notice(self._layout, self._hovered))
            self._pixels[self._hovered] = pixels
        faded = pixels if level >= 255 else (pixels.astype(numpy.uint16) * level // 255).astype(numpy.uint8)
        if hidden is not None:
            faded = faded.copy()
            faded[hidden[0]:hidden[1]] = 0
        self._api().push_frame(faded.tobytes())
        self._pushed = (self._hovered, level, hidden)

    def _on_mouse(self, is_down: bool, x: int, y: int) -> None:
        # Runs in the window procedure, in the middle of Tk's own event
        # dispatch. A call into Tk from here leaves _tkinter without its
        # thread state, and the next Tk callback aborts the process, so a
        # click is only noted here and the frame tick acts on it.
        if self._layout is None:
            return
        target = notice_target(self._layout, x, y)
        if is_down:
            self._pressed = target
            return
        pressed, self._pressed = self._pressed, None
        if target is not None and target == pressed:
            self._clicked = target
