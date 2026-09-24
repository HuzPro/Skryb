from collections.abc import Callable

from voicetotext.hotkeys.keys import HotkeyKey
from voicetotext.hotkeys.strategy import HotkeyStrategy


class ToggleStrategy(HotkeyStrategy):
    """Every press of the combo is one "toggle" request.

    Whether that starts or stops a recording is the use case's call, made
    from the app's real state. Keeping a start/stop flag here as well let it
    drift whenever a recording ended some other way - cancel, the pill's
    finish button, the length cap - and the next press was then wasted.
    """

    def __init__(self, keys: tuple[HotkeyKey, ...]) -> None:
        self._target_sequence = tuple(keys)
        self._seq_index: int = 0
        self._held_keys: set[HotkeyKey] = set()
        self._on_activate: Callable[[], None] | None = None
        self._on_deactivate: Callable[[], None] | None = None
        self._combo_already_triggered: bool = False

    def set_callbacks(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate

    def on_key_press(self, key: HotkeyKey) -> None:
        if (
            self._seq_index < len(self._target_sequence)
            and key == self._target_sequence[self._seq_index]
        ):
            self._seq_index += 1
            self._held_keys.add(key)
            if (
                self._seq_index == len(self._target_sequence)
                and not self._combo_already_triggered
            ):
                self._combo_already_triggered = True
                if self._on_activate:
                    self._on_activate()
        else:
            self._seq_index = 0
            self._held_keys.clear()

    def on_key_release(self, key: HotkeyKey) -> None:
        self._held_keys.discard(key)
        if not self._held_keys:
            self._seq_index = 0
            self._combo_already_triggered = False
