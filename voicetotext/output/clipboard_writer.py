from typing import Protocol, runtime_checkable


@runtime_checkable
class ClipboardWriter(Protocol):

    def set_text(self, text: str) -> None: ...
