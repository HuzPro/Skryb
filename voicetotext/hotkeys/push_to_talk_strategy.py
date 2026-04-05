from typing import Any, Callable

from voicetotext.hotkeys.key_parser import canonicalize_key
from voicetotext.hotkeys.strategy import HotkeyStrategy


class PushToTalkStrategy(HotkeyStrategy):

    def __init__(self, keys: set):
        self._target_keys = {canonicalize_key(k) for k in keys}
        self._pressed_keys: set = set()
        self._is_active = False
        self._on_activate: Callable[[], None] | None = None
        self._on_deactivate: Callable[[], None] | None = None

    def set_callbacks(self, on_activate, on_deactivate):
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate

    def on_key_press(self, key: Any) -> None:
        key = canonicalize_key(key)
        if key not in self._target_keys:
            return
        self._pressed_keys.add(key)
        if self._pressed_keys >= self._target_keys and not self._is_active:
            self._is_active = True
            if self._on_activate:
                self._on_activate()

    def on_key_release(self, key: Any) -> None:
        key = canonicalize_key(key)
        if key not in self._target_keys:
            return
        self._pressed_keys.discard(key)
        if self._is_active and not self._pressed_keys >= self._target_keys:
            self._is_active = False
            if self._on_deactivate:
                self._on_deactivate()
