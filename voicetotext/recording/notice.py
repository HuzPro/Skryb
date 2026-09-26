"""A small toast beside the dictation that never takes the keyboard focus."""

import ctypes
import tkinter
from collections.abc import Callable, Sequence
from typing import Protocol

from voicetotext.core.color_scheme import ColorScheme, detect_app_color_scheme
from voicetotext.core.work_area import foreground_work_area
from voicetotext.recording.pill import CANVAS_HEIGHT
from voicetotext.settings.theme import BODY_FONT, Theme, theme_for

NoticeAction = tuple[str, Callable[[], None]]

_AUTO_HIDE_MS = 10_000
_MAX_BUTTONS = 3
# The pill sits 26 px above the bottom of the work area; the notice sits
# just above the pill's full height, so the two never overlap.
_PILL_BOTTOM_MARGIN = 26
_GAP_ABOVE_PILL = 8
_PADDING = 12
_STRIP_WIDTH = 3

_GWL_EXSTYLE = -20
_WS_EX_TOPMOST = 0x00000008
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_NOACTIVATE = 0x08000000
_NO_ACTIVATE_STYLES = _WS_EX_TOPMOST | _WS_EX_TOOLWINDOW | _WS_EX_NOACTIVATE
_HWND_TOPMOST = -1
_SWP_NOACTIVATE = 0x0010
_SW_HIDE = 0
_SW_SHOWNOACTIVATE = 4


class NoticeWindowApi(Protocol):
    """The native half of the notice; tests swap in a fake."""

    def prepare(self, hwnd: int) -> None: ...
    def work_area(self) -> tuple[int, int, int, int]: ...
    def show(self, hwnd: int, x: int, y: int, width: int, height: int) -> None: ...
    def hide(self, hwnd: int) -> None: ...


class Win32NoticeWindow:
    """No-activate styles, placement on the foreground monitor, show and hide."""

    def __init__(self) -> None:
        user32 = ctypes.WinDLL("user32")
        user32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.GetWindowLongW.restype = ctypes.c_long
        user32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        user32.SetWindowLongW.restype = ctypes.c_long
        user32.GetForegroundWindow.argtypes = []
        user32.GetForegroundWindow.restype = ctypes.c_void_p
        user32.SetWindowPos.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint,
        ]
        user32.SetWindowPos.restype = ctypes.c_int
        user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        user32.ShowWindow.restype = ctypes.c_int
        self._user32 = user32

    def prepare(self, hwnd: int) -> None:
        style = self._user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        self._user32.SetWindowLongW(
            hwnd, _GWL_EXSTYLE, ctypes.c_long(style | _NO_ACTIVATE_STYLES).value,
        )

    def work_area(self) -> tuple[int, int, int, int]:
        return foreground_work_area()

    def show(self, hwnd: int, x: int, y: int, width: int, height: int) -> None:
        self._user32.SetWindowPos(
            hwnd, ctypes.c_void_p(_HWND_TOPMOST), x, y, width, height,
            _SWP_NOACTIVATE,
        )
        self._user32.ShowWindow(hwnd, _SW_SHOWNOACTIVATE)

    def hide(self, hwnd: int) -> None:
        self._user32.ShowWindow(hwnd, _SW_HIDE)


class Notice:
    """A one-line message with up to three buttons, just above the pill.

    GUI thread only; other threads hop over with the GUI dispatcher. The
    window is an ``overrideredirect`` toplevel, because Tk asks for the
    focus when it maps any other kind. It gets ``WS_EX_NOACTIVATE`` while
    still hidden and is shown with ``SW_SHOWNOACTIVATE``, so neither its
    appearance nor a click on it moves the focus off the app the text is
    going to. The buttons are classic Tk buttons: unlike ``ttk.Button``
    they have no click-to-focus binding. A new notice replaces the one
    showing, and each hides itself after ten seconds. A strip down the
    left edge is the accent colour, or amber when something went wrong.
    """

    def __init__(
        self,
        root: tkinter.Misc,
        window_api: NoticeWindowApi | None = None,
        *,
        color_scheme: Callable[[], ColorScheme] = detect_app_color_scheme,
    ) -> None:
        self._root = root
        self._window_api = window_api
        self._color_scheme = color_scheme
        self._window: tkinter.Toplevel | None = None
        self._hwnd = 0
        self._hide_after: str | None = None
        self._buttons: list[tkinter.Button] = []
        self._message = ""

    @property
    def visible(self) -> bool:
        return self._hide_after is not None

    @property
    def message(self) -> str:
        return self._message

    def buttons(self) -> list[tkinter.Button]:
        return list(self._buttons)

    def show(
        self, message: str, buttons: Sequence[NoticeAction] = (), *, warning: bool = False,
    ) -> None:
        window = self._window or self._build()
        self._cancel_auto_hide()
        self._fill(
            window, message, buttons[:_MAX_BUTTONS], theme_for(self._color_scheme()), warning,
        )
        window.update_idletasks()
        api = self._api()
        self._hwnd = int(window.wm_frame(), 16)
        api.prepare(self._hwnd)
        width, height = window.winfo_reqwidth(), window.winfo_reqheight()
        left, _top, right, bottom = api.work_area()
        api.show(
            self._hwnd,
            left + (right - left - width) // 2,
            bottom - _PILL_BOTTOM_MARGIN - CANVAS_HEIGHT - _GAP_ABOVE_PILL - height,
            width,
            height,
        )
        self._message = message
        self._hide_after = window.after(_AUTO_HIDE_MS, self.hide)

    def hide(self) -> None:
        if self._hide_after is None:
            return
        self._cancel_auto_hide()
        self._api().hide(self._hwnd)

    def dispose(self) -> None:
        self.hide()
        if self._window is not None:
            self._window.destroy()
            self._window = None

    def _build(self) -> tkinter.Toplevel:
        window = tkinter.Toplevel(self._root)
        window.withdraw()
        window.overrideredirect(True)
        self._window = window
        return window

    def _api(self) -> NoticeWindowApi:
        if self._window_api is None:
            self._window_api = Win32NoticeWindow()
        return self._window_api

    def _cancel_auto_hide(self) -> None:
        if self._hide_after is not None and self._window is not None:
            self._window.after_cancel(self._hide_after)
        self._hide_after = None

    def _fill(
        self,
        window: tkinter.Toplevel,
        message: str,
        actions: Sequence[NoticeAction],
        theme: Theme,
        warning: bool,
    ) -> None:
        for child in window.winfo_children():
            child.destroy()
        frame = tkinter.Frame(
            window, bg=theme.surface, highlightthickness=1,
            highlightbackground=theme.border, highlightcolor=theme.border,
        )
        frame.pack(fill="both", expand=True)
        tkinter.Frame(
            frame, bg=theme.warning if warning else theme.accent, width=_STRIP_WIDTH,
        ).pack(side="left", fill="y")
        tkinter.Label(
            frame, text=message, bg=theme.surface, fg=theme.foreground, font=BODY_FONT,
        ).pack(side="left", padx=(_PADDING, _PADDING + 4), pady=_PADDING)
        self._buttons = [
            self._button(frame, label, action, theme) for label, action in actions
        ]
        if self._buttons:
            self._buttons[-1].pack_configure(padx=(0, _PADDING))

    def _button(
        self,
        parent: tkinter.Frame,
        label: str,
        action: Callable[[], None],
        theme: Theme,
    ) -> tkinter.Button:
        button = tkinter.Button(
            parent, text=label, command=lambda: self._press(action),
            bg=theme.border, fg=theme.foreground,
            activebackground=theme.strong_border, activeforeground=theme.foreground,
            relief="flat", borderwidth=0, highlightthickness=0, takefocus=0,
            font=BODY_FONT, padx=12, pady=3, cursor="hand2",
        )
        button.pack(side="left", padx=(0, _PADDING // 2), pady=_PADDING)
        return button

    def _press(self, action: Callable[[], None]) -> None:
        self.hide()
        action()
