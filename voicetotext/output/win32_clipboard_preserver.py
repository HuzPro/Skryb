import ctypes
import ctypes.wintypes as _wt
import logging
from typing import Iterator

from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.win32_clipboard_open import open_clipboard_with_retry


_logger = logging.getLogger(__name__)

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32

_string_at = ctypes.string_at
_memmove = ctypes.memmove

_GMEM_MOVEABLE = 0x0002

# Standard clipboard formats that the Win32 docs guarantee use an
# ``HGLOBAL``-backed handle, which is what ``GlobalSize`` / ``GlobalLock``
# / ``_string_at`` operate on. Any other format may hand us an
# ``HBITMAP`` / ``HENHMETAFILE`` / ``HPALETTE`` / etc. — calling
# ``GlobalSize`` on those is undefined and can return garbage sizes that
# segfault ``_string_at`` and take the whole process with them. We use
# an *allowlist* and skip everything else.
#
# IDs and meanings come from the Windows ``CF_*`` constants in
# ``WinUser.h``.
_HGLOBAL_BACKED_STANDARD_FORMATS: frozenset[int] = frozenset({
    1,   # CF_TEXT
    4,   # CF_SYLK
    5,   # CF_DIF
    6,   # CF_TIFF
    7,   # CF_OEMTEXT
    8,   # CF_DIB
    10,  # CF_PENDATA
    11,  # CF_RIFF
    12,  # CF_WAVE
    13,  # CF_UNICODETEXT
    15,  # CF_HDROP
    16,  # CF_LOCALE
    17,  # CF_DIBV5
})
_REGISTERED_FORMAT_MIN = 0xC000


def _is_hglobal_backed(fmt: int) -> bool:
    return (
        fmt in _HGLOBAL_BACKED_STANDARD_FORMATS
        or fmt >= _REGISTERED_FORMAT_MIN
    )

# Install ctypes signatures. REQUIRED on 64-bit Python — without these,
# HANDLE (8-byte pointer) return values are truncated to c_int (4 bytes),
# silently corrupting clipboard handles and cascading into OverflowError
# when the bogus values flow into _string_at during save().
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
    if open_clipboard_with_retry(_user32):
        return True
    _logger.warning("Could not open clipboard for %s.", action)
    return False


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


def _allocate_and_copy(data: bytes) -> int | None:
    alloc = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(data))
    ptr = _kernel32.GlobalLock(alloc)
    if not ptr:
        _logger.warning("GlobalLock returned NULL during allocation, skipping.")
        return None
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
                if not _is_hglobal_backed(fmt):
                    _logger.debug(
                        "Skipping non-HGLOBAL clipboard format %d "
                        "(unsafe for raw GlobalSize/_string_at).", fmt,
                    )
                    continue
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
                handle = _allocate_and_copy(data)
                if handle is None:
                    continue
                _user32.SetClipboardData(fmt, handle)
        finally:
            _user32.CloseClipboard()
        self._snapshot.clear()
