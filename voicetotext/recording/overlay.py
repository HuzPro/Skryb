import ctypes
import ctypes.wintypes as wintypes
import math
import tkinter
from collections.abc import Callable
from typing import Protocol

import numpy
from PIL import Image, ImageDraw

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import DomainEvent, StateChangedEvent
from voicetotext.core.state_machine import AppState
from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher


_WIDTH = 128
_HEIGHT = 44
_BOTTOM_MARGIN = 36
_FRAME_INTERVAL_MS = 33
_SUPERSAMPLE = 4

_BAR_COUNT = 9
_BAR_WIDTH = 4
_BAR_GAP = 6
_BAR_MIN_HEIGHT = 4
_BAR_MAX_HEIGHT = 28

_BACKGROUND = (0x17, 0x19, 0x1D, 190)
_BORDER = (0x2A, 0x2E, 0x35, 190)
_BAR_COLOR = (0x70, 0xD6, 0xFF, 220)

_LEVEL_GAIN = 15.0
_LEVEL_SMOOTHING = 0.62
_PHASE_STEP = 0.32
_BREATHING_RATE = 0.18

_WINDOW_CLASS_NAME = "VoiceToTextRecordingOverlay"
_ERROR_CLASS_ALREADY_EXISTS = 1410
_WS_POPUP = 0x80000000
_WS_EX_LAYERED = 0x00080000
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_NOACTIVATE = 0x08000000
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_TOPMOST = 0x00000008
_SW_HIDE = 0
_SW_SHOWNOACTIVATE = 4
_HWND_TOPMOST = -1
_SWP_NOACTIVATE = 0x0010
_MONITOR_DEFAULTTONEAREST = 2
_BI_RGB = 0
_DIB_RGB_COLORS = 0
_AC_SRC_OVER = 0x00
_AC_SRC_ALPHA = 0x01
_ULW_ALPHA = 0x00000002

_LRESULT = ctypes.c_ssize_t
_WNDPROC = ctypes.WINFUNCTYPE(
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


def render_overlay_frame(level: float, phase: float) -> Image.Image:
    scale = _SUPERSAMPLE
    image = Image.new("RGBA", (_WIDTH * scale, _HEIGHT * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, (_WIDTH * scale) - 1, (_HEIGHT * scale) - 1),
        radius=(_HEIGHT * scale) / 2,
        fill=_BACKGROUND,
        outline=_BORDER,
        width=scale,
    )
    span = (_BAR_COUNT * _BAR_WIDTH) + ((_BAR_COUNT - 1) * _BAR_GAP)
    first_left = (_WIDTH - span) / 2
    for index in range(_BAR_COUNT):
        height = _bar_height(index, level, phase)
        left = (first_left + (index * (_BAR_WIDTH + _BAR_GAP))) * scale
        top = ((_HEIGHT - height) / 2) * scale
        draw.rounded_rectangle(
            (left, top, left + (_BAR_WIDTH * scale) - 1, top + (height * scale) - 1),
            radius=(_BAR_WIDTH * scale) / 2,
            fill=_BAR_COLOR,
        )
    return image.resize((_WIDTH, _HEIGHT), Image.Resampling.LANCZOS)


def premultiplied_bgra(image: Image.Image) -> bytes:
    rgba = numpy.asarray(image, dtype=numpy.uint8)
    alpha = rgba[:, :, 3]
    scaled = (rgba[:, :, :3].astype(numpy.uint16) * alpha[:, :, None]) // 255
    bgra = numpy.dstack((scaled[:, :, ::-1].astype(numpy.uint8), alpha))
    return numpy.ascontiguousarray(bgra).tobytes()


def _bar_height(index: int, level: float, phase: float) -> float:
    half = (_BAR_COUNT - 1) / 2
    distance = abs(index - half)
    bias = 1.0 - (0.62 * ((distance / half) ** 1.35))
    pulse = 0.55 + (0.45 * math.sin(phase + (distance * 0.8)))
    breathing = 0.06 + (0.06 * math.sin(phase * _BREATHING_RATE))
    amplitude = max(level * pulse, breathing) * bias
    return _BAR_MIN_HEIGHT + ((_BAR_MAX_HEIGHT - _BAR_MIN_HEIGHT) * amplitude)


class OverlayWindowApi(Protocol):
    def work_area_for_foreground(self) -> tuple[int, int, int, int]: ...
    def show_at(self, x: int, y: int) -> None: ...
    def push_frame(self, pixels: bytes) -> None: ...
    def hide(self) -> None: ...
    def dispose(self) -> None: ...


class Win32LayeredOverlayWindow:
    """A ``WS_EX_LAYERED`` popup painted straight from a premultiplied DIB.

    ``UpdateLayeredWindow`` is the only alpha source here on purpose:
    calling ``SetLayeredWindowAttributes`` on the same window switches it
    to the colour-key/constant-alpha path and the per-pixel alpha stops
    reaching the screen.
    """

    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32")
        self._gdi32 = ctypes.WinDLL("gdi32")
        self._kernel32 = ctypes.WinDLL("kernel32")
        self._declare_signatures()
        self._wnd_proc = _WNDPROC(
            ctypes.cast(self._user32.DefWindowProcW, ctypes.c_void_p).value,
        )
        self._register_class()
        self._hwnd = self._create_window()
        self._screen_dc: int = 0
        self._memory_dc: int = 0
        self._bitmap: int = 0
        self._previous_bitmap: int = 0
        self._bits = ctypes.c_void_p()

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
        window_class.lpfnWndProc = self._wnd_proc
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
            | _WS_EX_TRANSPARENT
            | _WS_EX_NOACTIVATE
            | _WS_EX_TOOLWINDOW
            | _WS_EX_TOPMOST,
            _WINDOW_CLASS_NAME,
            None,
            _WS_POPUP,
            0,
            0,
            _WIDTH,
            _HEIGHT,
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
        self._acquire_surface()
        self._user32.SetWindowPos(
            self._hwnd, _HWND_TOPMOST, x, y, _WIDTH, _HEIGHT, _SWP_NOACTIVATE,
        )
        self._user32.ShowWindow(self._hwnd, _SW_SHOWNOACTIVATE)

    def push_frame(self, pixels: bytes) -> None:
        if not self._bits:
            return
        ctypes.memmove(self._bits, pixels, len(pixels))
        size = wintypes.SIZE(_WIDTH, _HEIGHT)
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

    def hide(self) -> None:
        self._user32.ShowWindow(self._hwnd, _SW_HIDE)
        self._release_surface()

    def dispose(self) -> None:
        self.hide()
        if self._hwnd:
            self._user32.DestroyWindow(self._hwnd)
            self._hwnd = 0

    def _acquire_surface(self) -> None:
        if self._bitmap:
            return
        self._screen_dc = self._user32.GetDC(None)
        self._memory_dc = self._gdi32.CreateCompatibleDC(self._screen_dc)
        info = BitmapInfo()
        info.bmiHeader.biSize = ctypes.sizeof(BitmapInfoHeader)
        info.bmiHeader.biWidth = _WIDTH
        # Negative height selects a top-down DIB, matching PIL's row order.
        info.bmiHeader.biHeight = -_HEIGHT
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


class RecordingOverlay:
    def __init__(
        self,
        root: tkinter.Misc,
        event_bus: EventBus,
        dispatcher: GuiThreadDispatcher,
        level_provider: Callable[[], float],
        window_api: OverlayWindowApi | None = None,
    ) -> None:
        self._root = root
        self._event_bus = event_bus
        self._dispatcher = dispatcher
        self._level_provider = level_provider
        self._window_api = window_api or Win32LayeredOverlayWindow()
        self._level = 0.0
        self._phase = 0.0
        self._after_id: str | None = None
        self._event_bus.subscribe(StateChangedEvent, self._on_state_changed)

    def show(self) -> None:
        self._reset_level()
        left, _top, right, bottom = self._window_api.work_area_for_foreground()
        x = left + ((right - left - _WIDTH) // 2)
        y = bottom - _HEIGHT - _BOTTOM_MARGIN
        self._window_api.show_at(x, y)
        if self._after_id is None:
            self._animate()

    def hide(self) -> None:
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
            self._after_id = None
        self._reset_level()
        self._window_api.hide()

    def dispose(self) -> None:
        self._event_bus.unsubscribe(StateChangedEvent, self._on_state_changed)
        self.hide()
        self._window_api.dispose()

    def _on_state_changed(self, event: DomainEvent) -> None:
        if not isinstance(event, StateChangedEvent):
            return
        if event.new_state == AppState.RECORDING.value:
            self._dispatcher.dispatch(self.show)
        elif event.old_state == AppState.RECORDING.value:
            self._dispatcher.dispatch(self.hide)

    def _animate(self) -> None:
        normalized = min(1.0, max(0.0, self._level_provider() * _LEVEL_GAIN))
        self._level = (self._level * _LEVEL_SMOOTHING) + (
            normalized * (1.0 - _LEVEL_SMOOTHING)
        )
        self._phase += _PHASE_STEP
        frame = render_overlay_frame(self._level, self._phase)
        self._window_api.push_frame(premultiplied_bgra(frame))
        self._after_id = self._root.after(_FRAME_INTERVAL_MS, self._animate)

    def _reset_level(self) -> None:
        self._level = 0.0
        self._phase = 0.0
