"""The work area of the monitor the user is looking at."""

import ctypes
from functools import cache
from typing import Any

_MONITOR_DEFAULTTONEAREST = 2


class _MonitorInfo(ctypes.Structure):
    """``MONITORINFO`` in plain ctypes, so the module imports off Windows."""

    _fields_ = [
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", ctypes.c_long * 4),
        ("rcWork", ctypes.c_long * 4),
        ("dwFlags", ctypes.c_ulong),
    ]


def foreground_work_area() -> tuple[int, int, int, int]:
    """Left, top, right, bottom of the foreground window's monitor, taskbar excluded.

    Raises :class:`OSError` when Windows cannot say, and off Windows.
    """
    user32 = _user32()
    monitor = user32.MonitorFromWindow(
        user32.GetForegroundWindow(), _MONITOR_DEFAULTTONEAREST,
    )
    info = _MonitorInfo()
    info.cbSize = ctypes.sizeof(info)
    if not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        raise OSError("GetMonitorInfoW failed")
    left, top, right, bottom = info.rcWork
    return left, top, right, bottom


@cache
def _user32() -> Any:
    if not hasattr(ctypes, "WinDLL"):
        raise OSError("monitor work areas are a Windows feature")
    user32 = ctypes.WinDLL("user32")
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = ctypes.c_void_p
    user32.MonitorFromWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    user32.MonitorFromWindow.restype = ctypes.c_void_p
    user32.GetMonitorInfoW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MonitorInfo)]
    user32.GetMonitorInfoW.restype = ctypes.c_int
    return user32
