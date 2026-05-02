from collections.abc import Callable

from pynput.keyboard import Key, KeyCode

from voicetotext.hotkeys.key_parser import canonicalize_key
from voicetotext.hotkeys.strategy import HotkeyStrategy


class PushToTalkStrategy(HotkeyStrategy):

    def __init__(self, keys: tuple[Key | KeyCode, ...]) -> None:
        self._target_sequence = tuple(canonicalize_key(k) for k in keys)
        self._seq_index: int = 0
        self._held_keys: set[Key | KeyCode] = set()
        self._is_active: bool = False
        self._on_activate: Callable[[], None] | None = None
        self._on_deactivate: Callable[[], None] | None = None

    def set_callbacks(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate

    def on_key_press(self, key: Key | KeyCode) -> None:
        key = canonicalize_key(key)
        if self._is_active:
            return
        if (
            self._seq_index < len(self._target_sequence)
            and key == self._target_sequence[self._seq_index]
        ):
            self._seq_index += 1
            self._held_keys.add(key)
            if self._seq_index == len(self._target_sequence):
                self._is_active = True
                if self._on_activate:
                    self._on_activate()
        else:
            self._seq_index = 0
            self._held_keys.clear()

    def on_key_release(self, key: Key | KeyCode) -> None:
        key = canonicalize_key(key)
        self._held_keys.discard(key)
        if self._is_active:
            self._is_active = False
            self._seq_index = 0
            if self._on_deactivate:
                self._on_deactivate()
