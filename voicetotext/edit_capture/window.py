from dataclasses import dataclass


@dataclass(frozen=True)
class ForegroundWindow:
    """The window that had focus when an input happened.

    ``hwnd`` is 0 when the window could not be identified, which callers
    treat as ordinary data rather than an error.
    """

    hwnd: int = 0
    pid: int = 0
    process: str = ""
    title: str = ""
