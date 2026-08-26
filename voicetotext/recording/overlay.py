import ctypes
import ctypes.wintypes as wintypes
import math
import tkinter
from collections.abc import Callable
from typing import Protocol

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import DomainEvent, StateChangedEvent
from voicetotext.core.state_machine import AppState
from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher


_BAR_COUNT = 7
_FRAME_INTERVAL_MS = 33
_WIDTH = 112
_HEIGHT = 48
_BOTTOM_MARGIN = 36
_GWL_EXSTYLE = -20
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080
_LWA_ALPHA = 0x00000002
_MONITOR_DEFAULTTONEAREST = 2


class MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class OverlayWindowApi(Protocol):
    def work_area_for_foreground(self) -> tuple[int, int, int, int]: ...
    def make_click_through(self, hwnd: int) -> None: ...


class Win32OverlayWindowApi:
    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32")
        self._declare_signatures()

    def _declare_signatures(self) -> None:
        # Without these, ctypes narrows handles to C int and 64-bit HWNDs
        # arrive at the API truncated.
        self._user32.GetForegroundWindow.argtypes = []
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        self._user32.MonitorFromWindow.restype = wintypes.HANDLE
        self._user32.GetMonitorInfoW.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(MonitorInfo),
        ]
        self._user32.GetMonitorInfoW.restype = wintypes.BOOL
        self._user32.GetParent.argtypes = [wintypes.HWND]
        self._user32.GetParent.restype = wintypes.HWND
        self._user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.GetWindowLongW.restype = ctypes.c_long
        self._user32.SetWindowLongW.argtypes = [
            wintypes.HWND, ctypes.c_int, ctypes.c_long,
        ]
        self._user32.SetWindowLongW.restype = ctypes.c_long
        self._user32.SetLayeredWindowAttributes.argtypes = [
            wintypes.HWND, wintypes.COLORREF, wintypes.BYTE, wintypes.DWORD,
        ]
        self._user32.SetLayeredWindowAttributes.restype = wintypes.BOOL

    def work_area_for_foreground(self) -> tuple[int, int, int, int]:
        foreground = self._user32.GetForegroundWindow()
        monitor = self._user32.MonitorFromWindow(
            foreground, _MONITOR_DEFAULTTONEAREST,
        )
        info = MonitorInfo()
        info.cbSize = ctypes.sizeof(info)
        if not self._user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            raise OSError("GetMonitorInfoW failed")
        return (
            info.rcWork.left,
            info.rcWork.top,
            info.rcWork.right,
            info.rcWork.bottom,
        )

    def make_click_through(self, hwnd: int) -> None:
        parent = self._user32.GetParent(hwnd) or hwnd
        current = self._user32.GetWindowLongW(parent, _GWL_EXSTYLE)
        styles = (
            current
            | _WS_EX_LAYERED
            | _WS_EX_TRANSPARENT
            | _WS_EX_NOACTIVATE
            | _WS_EX_TOOLWINDOW
        )
        self._user32.SetWindowLongW(parent, _GWL_EXSTYLE, styles)
        if not self._user32.SetLayeredWindowAttributes(
            parent, 0, 225, _LWA_ALPHA,
        ):
            raise OSError("SetLayeredWindowAttributes failed")


class RecordingOverlay:
    def __init__(
        self,
        root: tkinter.Misc,
        event_bus: EventBus,
        dispatcher: GuiThreadDispatcher,
        level_provider: Callable[[], float],
        window_api: OverlayWindowApi | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._dispatcher = dispatcher
        self._level_provider = level_provider
        self._window_api = window_api or Win32OverlayWindowApi()
        self._level = 0.0
        self._phase = 0.0
        self._after_id: str | None = None
        self._window = tkinter.Toplevel(root)
        self._window.withdraw()
        self._window.overrideredirect(True)
        # Set before the first map: Tk rebuilds the wrapper window when this
        # attribute changes, which would discard the click-through styles.
        self._window.attributes("-topmost", True)
        self._window.configure(background="#17191d")
        self._canvas = tkinter.Canvas(
            self._window,
            width=_WIDTH,
            height=_HEIGHT,
            background="#17191d",
            highlightthickness=0,
            borderwidth=0,
        )
        self._canvas.pack()
        self._bars = [
            self._canvas.create_rectangle(0, 0, 0, 0, fill="#70d6ff", outline="")
            for _index in range(_BAR_COUNT)
        ]
        self._event_bus.subscribe(StateChangedEvent, self._on_state_changed)

    def show(self) -> None:
        self._reset_level()
        left, _top, right, bottom = self._window_api.work_area_for_foreground()
        x = left + ((right - left - _WIDTH) // 2)
        y = bottom - _HEIGHT - _BOTTOM_MARGIN
        self._window.geometry(f"{_WIDTH}x{_HEIGHT}+{x}+{y}")
        self._window.deiconify()
        self._window.update_idletasks()
        # The wrapper window only exists once Tk has mapped the toplevel, so
        # the styles have to be applied after ``deiconify``.
        self._window_api.make_click_through(self._window.winfo_id())
        if self._after_id is None:
            self._animate()

    def hide(self) -> None:
        if self._after_id is not None:
            self._window.after_cancel(self._after_id)
            self._after_id = None
        self._reset_level()
        self._window.withdraw()

    def dispose(self) -> None:
        self._event_bus.unsubscribe(StateChangedEvent, self._on_state_changed)
        self.hide()
        self._window.destroy()

    def _on_state_changed(self, event: DomainEvent) -> None:
        if not isinstance(event, StateChangedEvent):
            return
        if event.new_state == AppState.RECORDING.value:
            self._dispatcher.dispatch(self.show)
        elif event.old_state == AppState.RECORDING.value:
            self._dispatcher.dispatch(self.hide)

    def _animate(self) -> None:
        normalized = min(1.0, max(0.0, self._level_provider() * 12.0))
        self._level = (self._level * 0.72) + (normalized * 0.28)
        self._phase += 0.32
        for index, bar in enumerate(self._bars):
            pulse = 0.55 + (0.45 * math.sin(self._phase + (index * 0.8)))
            height = 6 + int((_HEIGHT - 14) * self._level * pulse)
            x1 = 13 + (index * 13)
            y1 = (_HEIGHT - height) // 2
            self._canvas.coords(bar, x1, y1, x1 + 7, y1 + height)
        self._after_id = self._window.after(
            _FRAME_INTERVAL_MS, self._animate,
        )

    def _reset_level(self) -> None:
        self._level = 0.0
        self._phase = 0.0
