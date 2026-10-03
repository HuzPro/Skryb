"""A small notice just above the pill, drawn the way the pill is, that never takes the focus."""

import tkinter
from collections.abc import Callable, Sequence
from typing import Protocol

from voicetotext.recording.overlay import (
    MouseHandler,
    Win32LayeredOverlayWindow,
    premultiplied_bgra,
)
from voicetotext.recording.pill import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_PILL_POSITION,
    NOTICE_CLOSE,
    NoticeLayout,
    PillPosition,
    notice_layout,
    notice_target,
    pill_origin,
    render_notice,
)

NoticeAction = tuple[str, Callable[[], None]]

_AUTO_HIDE_MS = 10_000
_HOVER_POLL_MS = 30
_MAX_BUTTONS = 3
_GAP_FROM_PILL = 8


class NoticeWindowApi(Protocol):
    """The native half of the notice; tests swap in a fake."""

    def work_area_for_foreground(self) -> tuple[int, int, int, int]: ...
    def resize(self, width: int, height: int) -> None: ...
    def show_at(self, x: int, y: int) -> None: ...
    def push_frame(self, pixels: bytes) -> None: ...
    def hide(self) -> None: ...
    def dispose(self) -> None: ...
    def set_mouse_handler(self, handler: MouseHandler) -> None: ...
    def cursor_position(self) -> tuple[int, int] | None: ...
    def set_hand_cursor(self, hand: bool) -> None: ...


def _default_position() -> PillPosition:
    return DEFAULT_PILL_POSITION


class Notice:
    """One message with up to three buttons and a close cross, just above the pill.

    GUI thread only; other threads hop over with the GUI dispatcher. It is
    a layered popup like the pill's own window, so it never takes the
    focus from the app the text is going to, and only its drawn shape
    takes clicks. It sits centred over the pill wherever the pill was
    dragged, or under it when the pill is at the top of the screen. A new
    notice replaces the one showing, and each hides itself after ten
    seconds.
    """

    def __init__(
        self,
        root: tkinter.Misc,
        window_factory: Callable[[], NoticeWindowApi] = Win32LayeredOverlayWindow,
        *,
        load_position: Callable[[], PillPosition] = _default_position,
    ) -> None:
        self._root = root
        self._window_factory = window_factory
        self._window: NoticeWindowApi | None = None
        self._load_position = load_position
        self._layout: NoticeLayout | None = None
        self._actions: list[Callable[[], None]] = []
        self._hovered: int | None = None
        self._pressed: int | None = None
        self._clicked: int | None = None
        self._hide_after: str | None = None
        self._poll_after: str | None = None
        self._message = ""

    @property
    def visible(self) -> bool:
        return self._hide_after is not None

    @property
    def message(self) -> str:
        return self._message

    @property
    def layout(self) -> NoticeLayout | None:
        return self._layout

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
        window = self._api()
        window.resize(layout.width, layout.height)
        window.show_at(*self._place(layout, window.work_area_for_foreground()))
        self._draw()
        self._hide_after = self._root.after(_AUTO_HIDE_MS, self.hide)
        self._poll_after = self._root.after(_HOVER_POLL_MS, self._follow_cursor)

    def hide(self) -> None:
        if self._hide_after is None:
            return
        self._root.after_cancel(self._hide_after)
        self._hide_after = None
        if self._poll_after is not None:
            self._root.after_cancel(self._poll_after)
            self._poll_after = None
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

    def _place(
        self, layout: NoticeLayout, work_area: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        left, top, right, _bottom = work_area
        pill_x, pill_y = pill_origin(self._load_position(), work_area)
        x = pill_x + (CANVAS_WIDTH - layout.width) // 2
        y = pill_y - _GAP_FROM_PILL - layout.height
        if y < top:
            y = pill_y + CANVAS_HEIGHT + _GAP_FROM_PILL
        return min(max(x, left), max(right - layout.width, left)), y

    def _draw(self) -> None:
        if self._layout is not None:
            self._api().push_frame(
                premultiplied_bgra(render_notice(self._layout, self._hovered)),
            )

    def _follow_cursor(self) -> None:
        self._poll_after = None
        if not self.visible or self._layout is None:
            return
        clicked, self._clicked = self._clicked, None
        if clicked is not None:
            self.hide()
            if clicked != NOTICE_CLOSE:
                self._actions[clicked]()
            return
        position = self._api().cursor_position()
        hovered = notice_target(self._layout, *position) if position is not None else None
        if hovered != self._hovered:
            self._hovered = hovered
            self._api().set_hand_cursor(hovered is not None)
            self._draw()
        self._poll_after = self._root.after(_HOVER_POLL_MS, self._follow_cursor)

    def _on_mouse(self, is_down: bool, x: int, y: int) -> None:
        # Runs in the window procedure, in the middle of Tk's own event
        # dispatch. A call into Tk from here leaves _tkinter without its
        # thread state, and the next Tk callback aborts the process, so a
        # click is only noted here and the cursor check acts on it.
        if self._layout is None:
            return
        target = notice_target(self._layout, x, y)
        if is_down:
            self._pressed = target
            return
        pressed, self._pressed = self._pressed, None
        if target is not None and target == pressed:
            self._clicked = target
