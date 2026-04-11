from typing import Callable

from pynput import keyboard

from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.strategy import HotkeyStrategy


class PynputAdapter(HotkeyListener):

    def __init__(self, strategy: HotkeyStrategy) -> None:
        self._strategy = strategy
        self._listener = None
        self._on_activate: Callable[[], None] | None = None
        self._on_deactivate: Callable[[], None] | None = None

    def start(self) -> None:
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def set_on_activate(self, callback: Callable[[], None]) -> None:
        self._on_activate = callback
        self._update_strategy_callbacks()

    def set_on_deactivate(self, callback: Callable[[], None]) -> None:
        self._on_deactivate = callback
        self._update_strategy_callbacks()

    def _update_strategy_callbacks(self) -> None:
        if self._on_activate is not None and self._on_deactivate is not None:
            self._strategy.set_callbacks(self._on_activate, self._on_deactivate)

    def _on_press(self, key: keyboard.Key) -> None:
        self._strategy.on_key_press(key)

    def _on_release(self, key: keyboard.Key) -> None:
        self._strategy.on_key_release(key)
