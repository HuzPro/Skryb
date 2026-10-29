"""Send a file or folder to the Windows Recycle Bin, so deleting it can be undone."""

import ctypes
from ctypes import wintypes
from pathlib import Path

_FO_DELETE = 3
_FOF_SILENT = 0x0004
_FOF_NOCONFIRMATION = 0x0010
_FOF_ALLOWUNDO = 0x0040
_FOF_NOERRORUI = 0x0400
# Where there is no Recycle Bin (a network drive, a file too big for the bin)
# Windows asks before deleting for good instead of doing it silently.
_FOF_WANTNUKEWARNING = 0x4000


class _FileOperation(ctypes.Structure):
    """SHFILEOPSTRUCTW; packed tight only on 32-bit Windows, which Skryb doesn't run on."""

    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        ("pFrom", wintypes.LPCWSTR),
        ("pTo", wintypes.LPCWSTR),
        ("fFlags", wintypes.WORD),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", wintypes.LPVOID),
        ("lpszProgressTitle", wintypes.LPCWSTR),
    ]


def move_to_recycle_bin(path: Path) -> None:
    """Raises OSError when it is still there afterwards: in use, refused, or kept when Windows asked."""
    operation = _FileOperation(
        wFunc=_FO_DELETE,
        # A list of paths ending in an empty one; ctypes adds the last terminator.
        pFrom=str(path.resolve()) + "\0",
        fFlags=_FOF_ALLOWUNDO | _FOF_NOCONFIRMATION | _FOF_SILENT | _FOF_NOERRORUI | _FOF_WANTNUKEWARNING,
    )
    code = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if operation.fAnyOperationsAborted:
        raise OSError(f"{path.name} was kept: deleting it was cancelled.")
    if code or path.exists():
        raise OSError(f"Windows couldn't move {path.name} to the Recycle Bin (code {code:#x}). Close anything using its files and try again.")
