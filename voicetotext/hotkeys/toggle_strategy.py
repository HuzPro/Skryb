from collections.abc import Callable

from voicetotext.hotkeys.keys import HotkeyKey
from voicetotext.hotkeys.strategy import ComboMatcher, HotkeyStrategy


def _nothing() -> None:
    return None


class ToggleStrategy(HotkeyStrategy):
    """Every press of the combo is one "toggle" request.

    Whether that starts or stops a recording is the use case's call, made
    from the app's real state. Keeping a start/stop flag here as well let it
    drift whenever a recording ended some other way - cancel, the pill's
    finish button, the length cap - and the next press was then wasted.
    Single-press shortcuts (paste or copy the last dictation) use it too.
    """

    def __init__(self, keys: tuple[HotkeyKey, ...]) -> None:
        self._matcher = ComboMatcher(keys)
        self._on_activate: Callable[[], None] = _nothing

    def set_callbacks(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        self._on_activate = on_activate

    def on_key_press(self, key: HotkeyKey) -> bool:
        if not self._matcher.press(key):
            return False
        self._on_activate()
        return True

    def on_key_release(self, key: HotkeyKey) -> None:
        self._matcher.release(key)
