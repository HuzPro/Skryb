from dataclasses import dataclass


@dataclass(frozen=True)
class ForegroundWindow:
    """Which window owned the keyboard, and which program drew it."""

    hwnd: int = 0
    pid: int = 0
    process: str = ""
    title: str = ""


# An hwnd of 0 means the window could not be identified. Callers treat it
# as ordinary data, so edit capture goes quiet rather than failing when
# the foreground window is elevated and refuses to be read.
UNKNOWN_WINDOW = ForegroundWindow()
