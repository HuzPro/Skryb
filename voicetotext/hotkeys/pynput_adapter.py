from typing import Callable

from pynput import keyboard

from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.strategy import HotkeyStrategy


class PynputAdapter(HotkeyListener):

    def __init__(self, strategy: HotkeyStrategy):
        self._strategy = strategy
        self._listener = None

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

    def _update_strategy_callbacks(self):
        activate = getattr(self, "_on_activate", None)
        deactivate = getattr(self, "_on_deactivate", None)
        if activate and deactivate:
            self._strategy.set_callbacks(activate, deactivate)

    def _on_press(self, key):
        self._strategy.on_key_press(key)

    def _on_release(self, key):
        self._strategy.on_key_release(key)
