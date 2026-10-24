from typing import Protocol, runtime_checkable

from skryb.output.paste_method import PasteMethod


@runtime_checkable
class KeySender(Protocol):
    """Injects the paste into the foreground window.

    Kept behind a port so the paste adapter stays free of any OS-input
    library and the concrete Win32 implementation can be swapped for a
    fake in tests. ``text`` is only typed for :attr:`PasteMethod.TYPE`;
    every other method pastes what is already on the clipboard.
    """

    def send_paste(self, method: PasteMethod, text: str) -> None: ...
