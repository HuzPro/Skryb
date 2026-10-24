from typing import Protocol

from skryb.output.sleeper import Sleeper


# OpenClipboard fails when another window currently holds the clipboard
# open. That is common during streaming dictation, where a delta is pasted
# every cadence while the user keeps typing elsewhere - the two race for the
# single global clipboard. The owner releases within a millisecond or two,
# so a short bounded retry turns a spurious "could not open clipboard"
# failure into a brief wait. The total budget is kept small (~20 ms) because
# this runs on the GUI thread.
_OPEN_ATTEMPTS = 5
_RETRY_SECONDS = 0.005


class _User32(Protocol):
    def OpenClipboard(self, hwnd: object) -> int: ...  # noqa: N802 (Win32 name)


def open_clipboard_with_retry(user32: _User32, sleeper: Sleeper) -> bool:
    """Open the clipboard, retrying briefly on contention.

    Returns ``True`` once ``OpenClipboard`` succeeds, or ``False`` if every
    attempt failed. Callers decide how to react (the preserver logs and
    degrades; the writer raises).
    """
    for attempt in range(_OPEN_ATTEMPTS):
        if user32.OpenClipboard(None):
            return True
        if attempt + 1 < _OPEN_ATTEMPTS:
            sleeper.sleep(_RETRY_SECONDS)
    return False
