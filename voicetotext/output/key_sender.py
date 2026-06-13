from typing import Protocol, runtime_checkable


@runtime_checkable
class KeySender(Protocol):
    """Injects a paste keystroke into the foreground window.

    Kept behind a port so the paste adapter stays free of any OS-input
    library and the concrete Win32 implementation can be swapped for a
    fake in tests.
    """

    def send_ctrl_v(self) -> None: ...
