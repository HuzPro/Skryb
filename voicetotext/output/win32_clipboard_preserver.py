import ctypes
import ctypes.wintypes
import logging

from voicetotext.output.clipboard_preserver import ClipboardPreserver


_logger = logging.getLogger(__name__)

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32

_string_at = ctypes.string_at
_memmove = ctypes.memmove

_GMEM_MOVEABLE = 0x0002


class Win32ClipboardPreserver(ClipboardPreserver):

    def __init__(self) -> None:
        self._snapshot: list[tuple[int, bytes]] = []

    def save(self) -> None:
        self._snapshot.clear()

        if not _user32.OpenClipboard(None):
            _logger.warning("Could not open clipboard for save.")
            return

        try:
            fmt = _user32.EnumClipboardFormats(0)
            while fmt != 0:
                handle = _user32.GetClipboardData(fmt)
                if handle is not None:
                    size = _kernel32.GlobalSize(handle)
                    ptr = _kernel32.GlobalLock(handle)
                    if ptr:
                        try:
                            data = _string_at(ptr, size)
                            self._snapshot.append((fmt, data))
                        finally:
                            _kernel32.GlobalUnlock(handle)
                    else:
                        _logger.warning(
                            "GlobalLock returned NULL for format %d, skipping.", fmt
                        )
                fmt = _user32.EnumClipboardFormats(fmt)
        finally:
            _user32.CloseClipboard()

    def restore(self) -> None:
        if not self._snapshot:
            return

        if not _user32.OpenClipboard(None):
            _logger.warning("Could not open clipboard for restore.")
            return

        try:
            _user32.EmptyClipboard()
            for fmt, data in self._snapshot:
                alloc = _kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(data))
                ptr = _kernel32.GlobalLock(alloc)
                try:
                    _memmove(ptr, data, len(data))
                finally:
                    _kernel32.GlobalUnlock(alloc)
                _user32.SetClipboardData(fmt, alloc)
        finally:
            _user32.CloseClipboard()

        self._snapshot.clear()
