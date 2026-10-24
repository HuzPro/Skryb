from collections.abc import Callable

from skryb.output.handler import TextOutputHandler


class CapsLockUppercaseDecorator(TextOutputHandler):
    """Types dictation in capitals while Caps Lock is on, as a keyboard would."""

    def __init__(
        self, inner: TextOutputHandler, caps_lock_enabled: Callable[[], bool],
    ) -> None:
        self._inner = inner
        self._caps_lock_enabled = caps_lock_enabled

    def output(self, text: str) -> None:
        self._inner.output(text.upper() if self._caps_lock_enabled() else text)
