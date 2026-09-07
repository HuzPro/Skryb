import ctypes
import ctypes.wintypes as _wt
import logging
import ntpath

from voicetotext.edit_capture.window import UNKNOWN_WINDOW, ForegroundWindow


_logger = logging.getLogger(__name__)

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WIN32_PATH_FORMAT = 0
_PATH_BUFFER_CHARS = 1024
_TITLE_BUFFER_CHARS = 512

_user32.GetForegroundWindow.argtypes = []
_user32.GetForegroundWindow.restype = _wt.HWND
_user32.GetWindowThreadProcessId.argtypes = [
    _wt.HWND, ctypes.POINTER(_wt.DWORD),
]
_user32.GetWindowThreadProcessId.restype = _wt.DWORD
_user32.GetWindowTextW.argtypes = [_wt.HWND, _wt.LPWSTR, ctypes.c_int]
_user32.GetWindowTextW.restype = ctypes.c_int
_kernel32.OpenProcess.argtypes = [_wt.DWORD, _wt.BOOL, _wt.DWORD]
_kernel32.OpenProcess.restype = _wt.HANDLE
_kernel32.QueryFullProcessImageNameW.argtypes = [
    _wt.HANDLE, _wt.DWORD, _wt.LPWSTR, ctypes.POINTER(_wt.DWORD),
]
_kernel32.QueryFullProcessImageNameW.restype = _wt.BOOL
_kernel32.CloseHandle.argtypes = [_wt.HANDLE]
_kernel32.CloseHandle.restype = _wt.BOOL


def foreground_hwnd() -> int:
    """The focused window's handle, without resolving its process.

    Edit capture asks this for every keystroke to decide whether the key
    belongs to the dictation it is watching, so it must stay to a single
    system call.
    """
    return int(_user32.GetForegroundWindow() or 0)


def foreground_window() -> ForegroundWindow:
    """Describe the window with the keyboard focus, or ``UNKNOWN_WINDOW``.

    Never raises: a locked workstation, a window owned by a more
    privileged process and a failed Win32 call all read as "unknown
    window", which callers handle as ordinary data.
    """
    try:
        return _read_foreground_window()
    except Exception:
        _logger.exception("Reading the foreground window failed.")
        return UNKNOWN_WINDOW


def _read_foreground_window() -> ForegroundWindow:
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return UNKNOWN_WINDOW
    pid = _wt.DWORD(0)
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return UNKNOWN_WINDOW
    process = _process_name(pid.value)
    if not process:
        return UNKNOWN_WINDOW
    return ForegroundWindow(
        hwnd=int(hwnd),
        pid=pid.value,
        process=process,
        title=_window_title(hwnd),
    )


def _process_name(pid: int) -> str:
    handle = _kernel32.OpenProcess(
        _PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
    )
    if not handle:
        return ""
    try:
        return _image_base_name(handle)
    finally:
        _kernel32.CloseHandle(handle)


def _image_base_name(handle: int) -> str:
    buffer = ctypes.create_unicode_buffer(_PATH_BUFFER_CHARS)
    size = _wt.DWORD(_PATH_BUFFER_CHARS)
    queried = _kernel32.QueryFullProcessImageNameW(
        handle, _WIN32_PATH_FORMAT, buffer, ctypes.byref(size),
    )
    if not queried:
        return ""
    return ntpath.basename(buffer.value).lower()


def _window_title(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(_TITLE_BUFFER_CHARS)
    _user32.GetWindowTextW(hwnd, buffer, _TITLE_BUFFER_CHARS)
    return buffer.value
