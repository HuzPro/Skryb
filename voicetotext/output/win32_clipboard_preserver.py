import ctypes
import ctypes.wintypes as _wt
import logging
from typing import Iterator

from voicetotext.output.clipboard_preserver import ClipboardPreserver


_logger = logging.getLogger(__name__)

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32

_string_at = ctypes.string_at
_memmove = ctypes.memmove

_GMEM_MOVEABLE = 0x0002

# Install ctypes signatures. REQUIRED on 64-bit Python — without these,
# HANDLE (8-byte pointer) return values are truncated to c_int (4 bytes),
# silently corrupting clipboard handles and cascading into OverflowError
# when the bogus values flow into _string_at during save().
#
# Compatible with pyperclip's signatures — both modules mutate the shared
# ctypes function objects from ctypes.windll, and pyperclip installs
# equivalent signatures lazily on its first copy() call. Installing them
# here at import time closes the timing window where save() could run
# against the uninitialized state.
_user32.OpenClipboard.argtypes = [_wt.HWND]
_user32.OpenClipboard.restype = _wt.BOOL

_user32.CloseClipboard.argtypes = []
_user32.CloseClipboard.restype = _wt.BOOL

_user32.EmptyClipboard.argtypes = []
_user32.EmptyClipboard.restype = _wt.BOOL

_user32.EnumClipboardFormats.argtypes = [_wt.UINT]
_user32.EnumClipboardFormats.restype = _wt.UINT

_user32.GetClipboardData.argtypes = [_wt.UINT]
_user32.GetClipboardData.restype = _wt.HANDLE

_user32.SetClipboardData.argtypes = [_wt.UINT, _wt.HANDLE]
_user32.SetClipboardData.restype = _wt.HANDLE

_kernel32.GlobalSize.argtypes = [_wt.HGLOBAL]
_kernel32.GlobalSize.restype = ctypes.c_size_t

_kernel32.GlobalLock.argtypes = [_wt.HGLOBAL]
_kernel32.GlobalLock.restype = ctypes.c_void_p

_kernel32.GlobalUnlock.argtypes = [_wt.HGLOBAL]
_kernel32.GlobalUnlock.restype = _wt.BOOL

_kernel32.GlobalAlloc.argtypes = [_wt.UINT, ctypes.c_size_t]
_kernel32.GlobalAlloc.restype = _wt.HGLOBAL


def _try_open_clipboard(action: str) -> bool:
    if not _user32.OpenClipboard(None):
        _logger.warning("Could not open clipboard for %s.", action)
        return False
    return True


def _iter_clipboard_formats() -> Iterator[int]:
    fmt = _user32.EnumClipboardFormats(0)
    while fmt != 0:
        yield fmt
        fmt = _user32.EnumClipboardFormats(fmt)


def _read_format_bytes(fmt: int) -> bytes | None:
    handle = _user32.GetClipboardData(fmt)
    if handle is None:
        return None
    size = _kernel32.GlobalSize(handle)
    ptr = _kernel32.GlobalLock(handle)
    if not ptr:
        _logger.warning("GlobalLock returned NULL for format %d, skipping.", fmt)
        return None
    try:
        return _string_at(ptr, size)
    finally:
        _kernel32.GlobalUnlock(handle)


def _allocate_and_copy(data: bytes) -> int:
    alloc = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(data))
    ptr = _kernel32.GlobalLock(alloc)
    try:
        _memmove(ptr, data, len(data))
    finally:
        _kernel32.GlobalUnlock(alloc)
    return alloc


class Win32ClipboardPreserver(ClipboardPreserver):

    def __init__(self) -> None:
        self._snapshot: list[tuple[int, bytes]] = []

    def save(self) -> None:
        self._snapshot.clear()
        if not _try_open_clipboard("save"):
            return
        try:
            for fmt in _iter_clipboard_formats():
                data = _read_format_bytes(fmt)
                if data is not None:
                    self._snapshot.append((fmt, data))
        finally:
            _user32.CloseClipboard()

    def restore(self) -> None:
        if not self._snapshot:
            return
        if not _try_open_clipboard("restore"):
            return
        try:
            _user32.EmptyClipboard()
            for fmt, data in self._snapshot:
                _user32.SetClipboardData(fmt, _allocate_and_copy(data))
        finally:
            _user32.CloseClipboard()
        self._snapshot.clear()
