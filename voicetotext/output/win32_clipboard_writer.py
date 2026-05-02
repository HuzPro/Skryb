import ctypes
import ctypes.wintypes as _wt

from voicetotext.output.clipboard_write_error import ClipboardWriteError


_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32
_memmove = ctypes.memmove

_GMEM_MOVEABLE = 0x0002
_CF_UNICODETEXT = 13

_user32.OpenClipboard.argtypes = [_wt.HWND]
_user32.OpenClipboard.restype = _wt.BOOL

_user32.CloseClipboard.argtypes = []
_user32.CloseClipboard.restype = _wt.BOOL

_user32.EmptyClipboard.argtypes = []
_user32.EmptyClipboard.restype = _wt.BOOL

_user32.SetClipboardData.argtypes = [_wt.UINT, _wt.HANDLE]
_user32.SetClipboardData.restype = _wt.HANDLE

_kernel32.GlobalAlloc.argtypes = [_wt.UINT, ctypes.c_size_t]
_kernel32.GlobalAlloc.restype = _wt.HGLOBAL

_kernel32.GlobalLock.argtypes = [_wt.HGLOBAL]
_kernel32.GlobalLock.restype = ctypes.c_void_p

_kernel32.GlobalUnlock.argtypes = [_wt.HGLOBAL]
_kernel32.GlobalUnlock.restype = _wt.BOOL


class Win32ClipboardWriter:

    def set_text(self, text: str) -> None:
        _write_text_to_clipboard(text)


def _write_text_to_clipboard(text: str) -> None:
    if not _user32.OpenClipboard(None):
        raise ClipboardWriteError("Could not open clipboard for write.")
    try:
        _user32.EmptyClipboard()
        encoded = (text + "\0").encode("utf-16-le")
        handle = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(encoded))
        if not handle:
            raise ClipboardWriteError("GlobalAlloc returned NULL.")
        ptr = _kernel32.GlobalLock(handle)
        if not ptr:
            raise ClipboardWriteError(
                "GlobalLock returned NULL; cannot write to clipboard."
            )
        try:
            _memmove(ptr, encoded, len(encoded))
        finally:
            _kernel32.GlobalUnlock(handle)
        if not _user32.SetClipboardData(_CF_UNICODETEXT, handle):
            raise ClipboardWriteError("SetClipboardData returned NULL.")
    finally:
        _user32.CloseClipboard()
