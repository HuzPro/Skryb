import ctypes
import ctypes.wintypes as wintypes
import logging
import math
import sys
import threading
import time
import tkinter
from collections.abc import Callable
from functools import cache
from typing import Protocol

import numpy
from PIL import Image

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    DomainEvent,
    MicrophoneAudioResumedEvent,
    MicrophoneSilentEvent,
    RecordingCapWarningEvent,
    RecordingPausedEvent,
    RecordingResumedEvent,
    RecordingStartedEvent,
    StateChangedEvent,
    TextPastedEvent,
)
from voicetotext.core.state_machine import AppState
from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.recording.input_devices import short_microphone_name
from voicetotext.recording.pill import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_PILL_POSITION,
    Button,
    Frame,
    Mode,
    PillPosition,
    Spring,
    VoiceHistory,
    collapsed_size,
    countdown_text,
    hit_test,
    is_settled_label,
    pill_origin,
    pill_position,
    render,
    settled_label_frame,
    shimmer_position,
    shimmered,
    target_size,
)

_logger = logging.getLogger(__name__)

# Each frame waits for the display's next refresh itself, so the timer
# only has to come back soon; a millisecond keeps up with 240 Hz.
_FRAME_POLL_MS = 1
# Without a refresh to wait on (no desktop compositor) a plain timer paces
# the frames instead, so the loop never spins.
_UNSYNCED_FRAME_MS = 8
_LONGEST_FRAME_SECONDS = 0.05
# While the pill is up, hand the GIL between threads more often, so a
# transcription busy in Python cannot hold the frame loop off for the
# default 5 ms, longer than a whole frame at 240 Hz. Measured against a
# thread that never leaves Python: frames every 1.2 ms (p95 1.3 ms) at
# 0.2 ms, against bursts of up to 27 ms at the default. That worst-case
# thread ran 13% slower meanwhile; a transcription spends most of its
# time in native code that lets the GIL go, so it pays far less.
_ANIMATING_SWITCH_INTERVAL = 0.0002
# Every call into Windows lets go of the GIL and has to win it back, which
# under a busy transcription can take a switch interval each time. The
# stray-click check costs three such calls, so it runs a few times a
# second rather than every frame.
_CLICK_CHECK_SECONDS = 0.05

# Two slices beyond the bars on show: one sliding in on the right, one
# fading out on the left.
_HISTORY_SLICES = 13
# Windows' own defaults for telling a drag from a wobbly click and a double
# click from two single ones.
_DRAG_THRESHOLD = 4
_DOUBLE_CLICK_SECONDS = 0.5
_DOUBLE_CLICK_SLACK = 4

_WINDOW_CLASS_NAME = "VoiceToTextRecordingOverlay"
_ERROR_CLASS_ALREADY_EXISTS = 1410
_WS_POPUP = 0x80000000
_WS_EX_LAYERED = 0x00080000
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_TOPMOST = 0x00000008
_SW_HIDE = 0
_SW_SHOWNOACTIVATE = 4
_HWND_TOPMOST = -1
_SWP_NOSIZE = 0x0001
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_MONITOR_DEFAULTTONEAREST = 2
_BI_RGB = 0
_DIB_RGB_COLORS = 0
_AC_SRC_OVER = 0x00
_AC_SRC_ALPHA = 0x01
_ULW_ALPHA = 0x00000002
_WM_SETCURSOR = 0x0020
_WM_MOUSEACTIVATE = 0x0021
_WM_LBUTTONDOWN = 0x0201
_WM_LBUTTONUP = 0x0202
_MA_NOACTIVATE = 3
_IDC_HAND = 32649
_VK_LBUTTON = 0x01
_VK_RBUTTON = 0x02
_SM_SWAPBUTTON = 23

_LRESULT = ctypes.c_ssize_t
# CFUNCTYPE stands in off Windows, so the pure drawing code imports on CI.
_WNDPROC = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
    _LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
)


class MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class WindowClass(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", _WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class BitmapInfoHeader(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BitmapInfo(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BitmapInfoHeader),
        ("bmiColors", wintypes.DWORD * 3),
    ]


class BlendFunction(ctypes.Structure):
    _fields_ = [
        ("BlendOp", wintypes.BYTE),
        ("BlendFlags", wintypes.BYTE),
        ("SourceConstantAlpha", wintypes.BYTE),
        ("AlphaFormat", wintypes.BYTE),
    ]


def premultiplied_array(image: Image.Image) -> numpy.ndarray:
    rgba = numpy.asarray(image, dtype=numpy.uint8)
    alpha = rgba[:, :, 3]
    scaled = (rgba[:, :, :3].astype(numpy.uint16) * alpha[:, :, None]) // 255
    bgra = numpy.dstack((scaled[:, :, ::-1].astype(numpy.uint8), alpha))
    return numpy.ascontiguousarray(bgra)


def premultiplied_bgra(image: Image.Image) -> bytes:
    return premultiplied_array(image).tobytes()


# Receives ``(is_button_down, x, y)`` in window pixels.
MouseHandler = Callable[[bool, int, int], None]


class OverlayWindowApi(Protocol):
    def work_area_for_foreground(self) -> tuple[int, int, int, int]: ...
    def show_at(self, x: int, y: int) -> None: ...
    def push_frame(self, pixels: bytes) -> None: ...
    def hide(self) -> None: ...
    def dispose(self) -> None: ...
    def set_mouse_handler(self, handler: MouseHandler) -> None: ...
    def cursor_position(self) -> tuple[int, int] | None: ...
    def set_hand_cursor(self, hand: bool) -> None: ...
    def wait_for_frame(self) -> bool: ...
    def move_to(self, x: int, y: int) -> None: ...
    def left_button_down(self) -> bool: ...


# Every overlay window shares one window class, and a class has one window
# procedure, so that procedure finds its window by handle.
_WINDOWS: dict[int, "Win32LayeredOverlayWindow"] = {}


def _window_proc(hwnd: int, message: int, w_param: int, l_param: int) -> int:
    window = _WINDOWS.get(int(hwnd or 0))
    handled = None
    if window is not None:
        try:
            handled = window.handle_message(message, l_param)
        except Exception:  # never raise into a Win32 callback
            _logger.exception("Overlay message %#x failed.", message)
    if handled is not None:
        return handled
    return _default_window_proc()(hwnd, message, w_param, l_param)


@cache
def _default_window_proc() -> Callable[[int, int, int, int], int]:
    user32 = ctypes.WinDLL("user32")
    user32.DefWindowProcW.argtypes = [
        wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
    ]
    user32.DefWindowProcW.restype = _LRESULT
    return user32.DefWindowProcW


# Pinned at module level: a ``WINFUNCTYPE`` the GC collects while Windows
# still holds its address would crash the interpreter.
_WINDOW_PROC = _WNDPROC(_window_proc)


def _client_point(l_param: int) -> tuple[int, int]:
    # Signed 16-bit halves: on a monitor left of the primary a coordinate
    # can be negative.
    x = ctypes.c_short(l_param & 0xFFFF).value
    y = ctypes.c_short((l_param >> 16) & 0xFFFF).value
    return x, y


class Win32LayeredOverlayWindow:
    """A ``WS_EX_LAYERED`` popup painted straight from a premultiplied DIB.

    ``UpdateLayeredWindow`` is the only alpha source here on purpose:
    calling ``SetLayeredWindowAttributes`` on the same window switches it
    to the colour-key/constant-alpha path and the per-pixel alpha stops
    reaching the screen.

    It takes clicks but never the focus: ``WS_EX_NOACTIVATE`` and
    ``MA_NOACTIVATE`` keep the caret in the app the dictation will paste
    into. Fully transparent pixels still pass clicks to the window
    beneath, so only the drawn pill is clickable.
    """

    # The pill's canvas unless resized; the notice above it uses one too.
    _size: tuple[int, int] = (CANVAS_WIDTH, CANVAS_HEIGHT)

    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32")
        self._gdi32 = ctypes.WinDLL("gdi32")
        self._kernel32 = ctypes.WinDLL("kernel32")
        self._declare_signatures()
        self._mouse_handler: MouseHandler | None = None
        self._hand_cursor = False
        self._origin = (0, 0)
        self._register_class()
        self._hwnd = self._create_window()
        _WINDOWS[self._hwnd] = self
        self._screen_dc: int = 0
        self._memory_dc: int = 0
        self._bitmap: int = 0
        self._previous_bitmap: int = 0
        self._bits = ctypes.c_void_p()
        self._pixels: numpy.ndarray | None = None

    def _declare_signatures(self) -> None:
        # Without these, ctypes narrows handles to C int and 64-bit HWNDs
        # arrive at the API truncated.
        self._kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        self._kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        self._user32.GetForegroundWindow.argtypes = []
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
        self._user32.MonitorFromWindow.restype = wintypes.HANDLE
        self._user32.GetMonitorInfoW.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(MonitorInfo),
        ]
        self._user32.GetMonitorInfoW.restype = wintypes.BOOL
        self._user32.RegisterClassW.argtypes = [ctypes.POINTER(WindowClass)]
        self._user32.RegisterClassW.restype = wintypes.ATOM
        self._user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        self._user32.CreateWindowExW.restype = wintypes.HWND
        self._user32.DestroyWindow.argtypes = [wintypes.HWND]
        self._user32.DestroyWindow.restype = wintypes.BOOL
        self._user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.ShowWindow.restype = wintypes.BOOL
        self._user32.SetWindowPos.argtypes = [
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        ]
        self._user32.SetWindowPos.restype = wintypes.BOOL
        self._user32.GetDC.argtypes = [wintypes.HWND]
        self._user32.GetDC.restype = wintypes.HDC
        self._user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        self._user32.ReleaseDC.restype = ctypes.c_int
        self._user32.UpdateLayeredWindow.argtypes = [
            wintypes.HWND,
            wintypes.HDC,
            ctypes.POINTER(wintypes.POINT),
            ctypes.POINTER(wintypes.SIZE),
            wintypes.HDC,
            ctypes.POINTER(wintypes.POINT),
            wintypes.COLORREF,
            ctypes.POINTER(BlendFunction),
            wintypes.DWORD,
        ]
        self._user32.UpdateLayeredWindow.restype = wintypes.BOOL
        self._user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
        self._user32.GetCursorPos.restype = wintypes.BOOL
        self._user32.LoadCursorW.argtypes = [wintypes.HINSTANCE, wintypes.LPVOID]
        self._user32.LoadCursorW.restype = wintypes.HANDLE
        self._user32.SetCursor.argtypes = [wintypes.HANDLE]
        self._user32.SetCursor.restype = wintypes.HANDLE
        self._user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self._user32.GetAsyncKeyState.restype = ctypes.c_short
        self._user32.GetSystemMetrics.argtypes = [ctypes.c_int]
        self._user32.GetSystemMetrics.restype = ctypes.c_int
        self._dwmapi = ctypes.WinDLL("dwmapi")
        self._dwmapi.DwmFlush.argtypes = []
        self._dwmapi.DwmFlush.restype = ctypes.c_long
        self._winmm = ctypes.WinDLL("winmm")
        self._winmm.timeBeginPeriod.argtypes = [wintypes.UINT]
        self._winmm.timeEndPeriod.argtypes = [wintypes.UINT]
        self._gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        self._gdi32.CreateCompatibleDC.restype = wintypes.HDC
        self._gdi32.CreateDIBSection.argtypes = [
            wintypes.HDC,
            ctypes.POINTER(BitmapInfo),
            wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p),
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        self._gdi32.CreateDIBSection.restype = wintypes.HBITMAP
        self._gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        self._gdi32.SelectObject.restype = wintypes.HGDIOBJ
        self._gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        self._gdi32.DeleteObject.restype = wintypes.BOOL
        self._gdi32.DeleteDC.argtypes = [wintypes.HDC]
        self._gdi32.DeleteDC.restype = wintypes.BOOL

    def _register_class(self) -> None:
        window_class = WindowClass()
        window_class.lpfnWndProc = _WINDOW_PROC
        window_class.hInstance = self._kernel32.GetModuleHandleW(None)
        window_class.lpszClassName = _WINDOW_CLASS_NAME
        if self._user32.RegisterClassW(ctypes.byref(window_class)):
            return
        error = self._kernel32.GetLastError()
        if error != _ERROR_CLASS_ALREADY_EXISTS:
            raise OSError(f"RegisterClassW failed ({error})")

    def _create_window(self) -> int:
        hwnd = self._user32.CreateWindowExW(
            _WS_EX_LAYERED
            | _WS_EX_NOACTIVATE
            | _WS_EX_TOOLWINDOW
            | _WS_EX_TOPMOST,
            _WINDOW_CLASS_NAME,
            None,
            _WS_POPUP,
            0,
            0,
            *self._size,
            None,
            None,
            self._kernel32.GetModuleHandleW(None),
            None,
        )
        if not hwnd:
            raise OSError("CreateWindowExW failed")
        return int(hwnd)

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

    def show_at(self, x: int, y: int) -> None:
        if not self._bitmap:
            # Windows timers tick every 15.6 ms by default, which would
            # round the frame timer up to a quarter of the frames 240 Hz
            # needs. Asked for only while the pill is on screen.
            self._winmm.timeBeginPeriod(1)
        self._acquire_surface()
        self._origin = (x, y)
        self._user32.SetWindowPos(
            self._hwnd, _HWND_TOPMOST, x, y, *self._size,
            _SWP_NOACTIVATE,
        )
        self._user32.ShowWindow(self._hwnd, _SW_SHOWNOACTIVATE)

    def push_frame(self, pixels: bytes) -> None:
        if self._pixels is None:
            return
        # A numpy copy into the DIB stays inside Python, where memmove
        # would give the GIL away and have to wait to win it back.
        self._pixels[:] = numpy.frombuffer(pixels, dtype=numpy.uint8)
        size = wintypes.SIZE(*self._size)
        source = wintypes.POINT(0, 0)
        blend = BlendFunction(_AC_SRC_OVER, 0, 255, _AC_SRC_ALPHA)
        if not self._user32.UpdateLayeredWindow(
            self._hwnd,
            self._screen_dc,
            None,
            ctypes.byref(size),
            self._memory_dc,
            ctypes.byref(source),
            0,
            ctypes.byref(blend),
            _ULW_ALPHA,
        ):
            raise OSError("UpdateLayeredWindow failed")

    def resize(self, width: int, height: int) -> None:
        """Takes effect from the next ``show_at``; a window showing is hidden first."""
        if (width, height) != self._size:
            self.hide()
            self._size = (width, height)

    def hide(self) -> None:
        self._user32.ShowWindow(self._hwnd, _SW_HIDE)
        if self._bitmap:
            self._winmm.timeEndPeriod(1)
        self._release_surface()

    def wait_for_frame(self) -> bool:
        """Block until the compositor's next refresh; False if it has none.

        ctypes lets go of the GIL for the wait, so the transcription keeps
        running while the frame loop sleeps.
        """
        return self._dwmapi.DwmFlush() == 0

    def dispose(self) -> None:
        self.hide()
        if self._hwnd:
            _WINDOWS.pop(self._hwnd, None)
            self._user32.DestroyWindow(self._hwnd)
            self._hwnd = 0

    def set_mouse_handler(self, handler: MouseHandler) -> None:
        self._mouse_handler = handler

    def cursor_position(self) -> tuple[int, int] | None:
        point = wintypes.POINT()
        if not self._user32.GetCursorPos(ctypes.byref(point)):
            return None
        return point.x - self._origin[0], point.y - self._origin[1]

    def set_hand_cursor(self, hand: bool) -> None:
        self._hand_cursor = hand

    def move_to(self, x: int, y: int) -> None:
        self._origin = (x, y)
        self._user32.SetWindowPos(
            self._hwnd, None, x, y, 0, 0,
            _SWP_NOACTIVATE | _SWP_NOSIZE | _SWP_NOZORDER,
        )

    def left_button_down(self) -> bool:
        """Whether the primary button is held, wherever the cursor is.

        A release off the pill goes to the window under it, so a drag also
        watches the button itself. ``GetAsyncKeyState`` reads the physical
        buttons, and with them swapped the primary one is on the right.
        """
        swapped = self._user32.GetSystemMetrics(_SM_SWAPBUTTON)
        button = _VK_RBUTTON if swapped else _VK_LBUTTON
        return bool(self._user32.GetAsyncKeyState(button) & 0x8000)

    def handle_message(self, message: int, l_param: int) -> int | None:
        """The window procedure's answer, or None for the default one."""
        if message == _WM_MOUSEACTIVATE:
            return _MA_NOACTIVATE
        if message == _WM_SETCURSOR and self._hand_cursor:
            self._user32.SetCursor(self._user32.LoadCursorW(None, _IDC_HAND))
            return 1
        if message in (_WM_LBUTTONDOWN, _WM_LBUTTONUP):
            if self._mouse_handler is not None:
                x, y = _client_point(l_param)
                self._mouse_handler(message == _WM_LBUTTONDOWN, x, y)
            return 0
        return None

    def _acquire_surface(self) -> None:
        if self._bitmap:
            return
        self._screen_dc = self._user32.GetDC(None)
        self._memory_dc = self._gdi32.CreateCompatibleDC(self._screen_dc)
        info = BitmapInfo()
        info.bmiHeader.biSize = ctypes.sizeof(BitmapInfoHeader)
        width, height = self._size
        info.bmiHeader.biWidth = width
        # Negative height selects a top-down DIB, matching PIL's row order.
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = _BI_RGB
        self._bitmap = self._gdi32.CreateDIBSection(
            self._screen_dc,
            ctypes.byref(info),
            _DIB_RGB_COLORS,
            ctypes.byref(self._bits),
            None,
            0,
        )
        if not self._bitmap:
            raise OSError("CreateDIBSection failed")
        self._pixels = numpy.ctypeslib.as_array(
            ctypes.cast(self._bits, ctypes.POINTER(ctypes.c_uint8)),
            shape=(width * height * 4,),
        )
        self._previous_bitmap = self._gdi32.SelectObject(
            self._memory_dc, self._bitmap,
        )

    def _release_surface(self) -> None:
        if not self._bitmap:
            return
        self._gdi32.SelectObject(self._memory_dc, self._previous_bitmap)
        self._gdi32.DeleteObject(self._bitmap)
        self._gdi32.DeleteDC(self._memory_dc)
        self._user32.ReleaseDC(None, self._screen_dc)
        self._bitmap = 0
        self._memory_dc = 0
        self._screen_dc = 0
        self._bits = ctypes.c_void_p()
        self._pixels = None


def _ignore() -> None:
    return None


def _default_position() -> PillPosition:
    return DEFAULT_PILL_POSITION


def _forget_position(_position: PillPosition) -> None:
    return None


class RecordingOverlay:
    """The pill: grows in on record, shimmers while transcribing, leaves as the text is pasted.

    Everything here runs on the Tk thread. Events arrive on worker threads
    and are handed over through the dispatcher; clicks arrive on the Tk
    thread already, through the window procedure.

    Dragging the pill by anything but its buttons moves it, and it comes
    back there next time; a double click puts it back at the bottom centre.
    """

    def __init__(
        self,
        root: tkinter.Misc,
        event_bus: EventBus,
        dispatcher: GuiThreadDispatcher,
        level_provider: Callable[[], float],
        window_api: OverlayWindowApi | None = None,
        on_cancel: Callable[[], None] = _ignore,
        on_pause_toggle: Callable[[], None] = _ignore,
        on_finish: Callable[[], None] = _ignore,
        on_press: Callable[[], None] = _ignore,
        on_tick: Callable[[], None] = _ignore,
        clock: Callable[[], float] = time.monotonic,
        load_position: Callable[[], PillPosition] = _default_position,
        save_position: Callable[[PillPosition], None] = _forget_position,
    ) -> None:
        self._root = root
        self._event_bus = event_bus
        self._dispatcher = dispatcher
        self._level_provider = level_provider
        self._window_api = window_api or Win32LayeredOverlayWindow()
        self._actions = {
            Button.CANCEL: on_cancel,
            Button.PAUSE: on_pause_toggle,
            Button.FINISH: on_finish,
        }
        self._on_press = on_press
        self._on_tick = on_tick
        self._clock = clock
        self._load_position = load_position
        self._save_position = save_position
        width, height = collapsed_size()
        self._width = Spring(width)
        self._height = Spring(height)
        # Fades never overshoot: a bounce in opacity reads as a flicker.
        self._opacity = Spring(0.0, damping=40.0)
        self._controls = Spring(0.0, damping=40.0)
        self._label = Spring(0.0, damping=40.0)
        self._mode = Mode.CLOSED
        self._visible = False
        self._paused = False
        # Kept across recordings: it has learnt how loud this speaker is.
        self._voice = VoiceHistory(_HISTORY_SLICES)
        self._transcribing_since = 0.0
        self._settled_label = premultiplied_array(render(settled_label_frame()))
        self._switch_interval: float | None = None
        self._pressed: Button | None = None
        self._caption = ""
        self._work_area = (0, 0, 0, 0)
        self._origin = (0, 0)
        # Where on the pill a press that may become a drag landed.
        self._grab: tuple[int, int] | None = None
        self._dragging = False
        self._last_click: tuple[float, int, int] | None = None
        self._silent_microphone = ""
        self._countdown: float | None = None
        self._frame = self._current_frame(None)
        self._last_tick = 0.0
        self._last_click_check = float("-inf")
        self._after_id: str | None = None
        self._window_api.set_mouse_handler(self._on_mouse)
        self._subscriptions: tuple[
            tuple[type[DomainEvent], Callable[[DomainEvent], None]], ...
        ] = (
            (StateChangedEvent, self._on_state_changed),
            (RecordingPausedEvent, self._on_paused),
            (RecordingResumedEvent, self._on_resumed),
            (TextPastedEvent, self._on_pasted),
            (MicrophoneSilentEvent, self._on_microphone_silent),
            (MicrophoneAudioResumedEvent, self._on_microphone_back),
            (RecordingCapWarningEvent, self._on_cap_warning),
            (RecordingStartedEvent, self._on_recording_started),
        )
        for event_type, handler in self._subscriptions:
            self._event_bus.subscribe(event_type, handler)

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def frame(self) -> Frame:
        return self._frame

    def show(self) -> None:
        """Grow the pill in for a new recording."""
        if not self._visible:
            self._work_area = self._window_api.work_area_for_foreground()
            self._origin = pill_origin(self._load_position(), self._work_area)
            self._window_api.show_at(*self._origin)
            self._visible = True
            self._switch_interval = sys.getswitchinterval()
            sys.setswitchinterval(_ANIMATING_SWITCH_INTERVAL)
            self._opacity.value = 0.0
            self._width.value, self._height.value = collapsed_size()
        self._mode = Mode.RECORDING
        self._paused = False
        self._voice.clear()
        self._silent_microphone = ""
        self._countdown = None
        self._retarget()
        self._start_loop()

    def hide(self) -> None:
        """Shrink the pill away; the window hides once it has faded out."""
        if not self._visible:
            return
        self._mode = Mode.CLOSED
        self._pressed = None
        if self._grab is not None:
            self._release_grab()
        self._retarget()
        self._start_loop()

    def dispose(self) -> None:
        for event_type, handler in self._subscriptions:
            self._event_bus.unsubscribe(event_type, handler)
        self._stop_loop()
        self._restore_switch_interval()
        self._visible = False
        self._window_api.dispose()

    def _start_transcribing(self) -> None:
        if not self._visible:
            return
        self._mode = Mode.TRANSCRIBING
        self._pressed = None
        self._transcribing_since = self._clock()
        self._retarget()
        self._start_loop()

    def _finish_on_paste(self) -> None:
        if self._mode is Mode.TRANSCRIBING:
            self.hide()

    def _set_paused(self, paused: bool) -> None:
        self._paused = paused

    def _set_caption(self, caption: str) -> None:
        self._caption = caption

    def _set_silent_microphone(self, name: str) -> None:
        self._silent_microphone = name

    def _set_countdown(self, seconds: float) -> None:
        self._countdown = seconds

    def _retarget(self) -> None:
        width, height = target_size(self._mode)
        self._width.target = width
        self._height.target = height
        self._opacity.target = 0.0 if self._mode is Mode.CLOSED else 1.0
        self._controls.target = 1.0 if self._mode is Mode.RECORDING else 0.0
        self._label.target = 1.0 if self._mode is Mode.TRANSCRIBING else 0.0

    def _on_state_changed(self, event: DomainEvent) -> None:
        if not isinstance(event, StateChangedEvent):
            return
        old, new = event.old_state, event.new_state
        if new == AppState.RECORDING.value:
            self._dispatcher.dispatch(self.show)
        elif old == AppState.RECORDING.value and new == AppState.TRANSCRIBING.value:
            self._dispatcher.dispatch(self._start_transcribing)
        elif old == AppState.TRANSCRIBING.value:
            # Usually the paste has closed it already; this covers an empty
            # result, a clipboard-only mode and a failure.
            self._dispatcher.dispatch(self.hide)
        elif old == AppState.RECORDING.value:
            self._dispatcher.dispatch(self.hide)

    def _on_paused(self, _event: DomainEvent) -> None:
        self._dispatcher.dispatch(lambda: self._set_paused(True))

    def _on_resumed(self, _event: DomainEvent) -> None:
        self._dispatcher.dispatch(lambda: self._set_paused(False))

    def _on_pasted(self, _event: DomainEvent) -> None:
        # Published on the GUI thread in the middle of the paste, which still
        # has the clipboard restore to wait out: queued, the pill would only
        # start leaving after it. Run now, so it leaves as the text lands.
        if threading.current_thread() is threading.main_thread():
            self._finish_on_paste()
        else:
            self._dispatcher.dispatch(self._finish_on_paste)

    def _on_recording_started(self, event: DomainEvent) -> None:
        if isinstance(event, RecordingStartedEvent):
            caption = event.caption
            self._dispatcher.dispatch(lambda: self._set_caption(caption))

    def _on_microphone_silent(self, event: DomainEvent) -> None:
        if isinstance(event, MicrophoneSilentEvent):
            name = short_microphone_name(event.device_name)
            self._dispatcher.dispatch(lambda: self._set_silent_microphone(name))

    def _on_microphone_back(self, _event: DomainEvent) -> None:
        self._dispatcher.dispatch(lambda: self._set_silent_microphone(""))

    def _on_cap_warning(self, event: DomainEvent) -> None:
        if isinstance(event, RecordingCapWarningEvent):
            remaining = event.remaining_seconds
            self._dispatcher.dispatch(lambda: self._set_countdown(remaining))

    def _on_mouse(self, is_down: bool, x: int, y: int) -> None:
        button = hit_test(self._frame, x, y)
        if is_down:
            self._on_press()
            self._pressed = button
            if button is None:
                self._grab = (x, y)
                self._dragging = False
            return
        pressed, self._pressed = self._pressed, None
        if self._grab is not None:
            self._release_grab()
        elif button is not None and button is pressed:
            self._actions[button]()

    def _follow_drag(self) -> None:
        if self._grab is None:
            return
        if not self._window_api.left_button_down():
            self._release_grab()
            return
        cursor = self._window_api.cursor_position()
        if cursor is None:
            return
        dx, dy = cursor[0] - self._grab[0], cursor[1] - self._grab[1]
        if not self._dragging and math.hypot(dx, dy) < _DRAG_THRESHOLD:
            return
        self._dragging = True
        wanted = (self._origin[0] + dx, self._origin[1] + dy)
        self._move_to(pill_origin(pill_position(wanted, self._work_area), self._work_area))

    def _release_grab(self) -> None:
        grab, self._grab = self._grab, None
        if self._dragging:
            self._dragging = False
            self._last_click = None
            self._save_position(pill_position(self._origin, self._work_area))
            return
        if grab is None:
            return
        now = self._clock()
        last, self._last_click = self._last_click, (now, *grab)
        if (
            last is not None
            and now - last[0] <= _DOUBLE_CLICK_SECONDS
            and math.hypot(grab[0] - last[1], grab[1] - last[2]) <= _DOUBLE_CLICK_SLACK
        ):
            self._last_click = None
            self._move_to(pill_origin(DEFAULT_PILL_POSITION, self._work_area))
            self._save_position(DEFAULT_PILL_POSITION)

    def _move_to(self, origin: tuple[int, int]) -> None:
        if origin != self._origin:
            self._origin = origin
            self._window_api.move_to(*origin)

    def _start_loop(self) -> None:
        if self._after_id is None:
            self._last_tick = self._clock()
            self._tick()

    def _stop_loop(self) -> None:
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
            self._after_id = None

    def _tick(self) -> None:
        self._after_id = None
        synced = self._window_api.wait_for_frame()
        now = self._clock()
        elapsed = max(now - self._last_tick, 0.0)
        seconds = min(elapsed, _LONGEST_FRAME_SECONDS)
        self._last_tick = now
        for spring in self._springs():
            spring.step(seconds)
        self._advance_level(seconds)
        self._advance_countdown(elapsed)
        self._follow_drag()
        if now - self._last_click_check >= _CLICK_CHECK_SECONDS:
            self._last_click_check = now
            self._on_tick()
        hovered = self._hovered()
        self._window_api.set_hand_cursor(hovered is not None)
        self._frame = self._current_frame(hovered)
        base = (
            self._settled_label if is_settled_label(self._frame)
            else premultiplied_array(render(self._frame))
        )
        self._window_api.push_frame(shimmered(base, self._frame).tobytes())
        if self._mode is Mode.CLOSED and self._opacity.value < 0.01:
            self._visible = False
            self._window_api.hide()
            self._restore_switch_interval()
            return
        self._after_id = self._root.after(
            _FRAME_POLL_MS if synced else _UNSYNCED_FRAME_MS, self._tick,
        )

    def _restore_switch_interval(self) -> None:
        if self._switch_interval is not None:
            sys.setswitchinterval(self._switch_interval)
            self._switch_interval = None

    def _springs(self) -> tuple[Spring, ...]:
        return self._width, self._height, self._opacity, self._controls, self._label

    def _advance_level(self, seconds: float) -> None:
        # Paused, the row holds still rather than scrolling silence past.
        if self._mode is Mode.RECORDING and not self._paused:
            self._voice.update(self._level_provider(), seconds)

    def _advance_countdown(self, elapsed: float) -> None:
        # Paused time does not count towards the limit, so the clock stops too.
        if self._countdown is not None and self._mode is Mode.RECORDING and not self._paused:
            self._countdown = max(self._countdown - elapsed, 0.0)

    def _hovered(self) -> Button | None:
        if self._controls.value < 0.9:
            return None
        position = self._window_api.cursor_position()
        if position is None:
            return None
        return hit_test(self._frame, *position)

    def _current_frame(self, hovered: Button | None) -> Frame:
        return Frame(
            width=max(self._width.value, 0.0),
            height=max(self._height.value, 0.0),
            opacity=_unit(self._opacity.value),
            controls=_unit(self._controls.value),
            label=_unit(self._label.value),
            levels=self._voice.levels(),
            scroll=self._voice.scroll(),
            paused=self._paused,
            shimmer=(
                shimmer_position(self._clock() - self._transcribing_since)
                if self._mode is Mode.TRANSCRIBING else -1.0
            ),
            hovered=hovered,
            caption=self._caption,
            silent_microphone=self._silent_microphone,
            countdown=(
                countdown_text(self._countdown) if self._countdown is not None else ""
            ),
        )


def _unit(value: float) -> float:
    return min(max(value, 0.0), 1.0)
