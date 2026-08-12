from collections.abc import Callable

from voicetotext.hotkeys import keys
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.strategy import HotkeyStrategy
from voicetotext.hotkeys.win32_keyboard_hook import Win32KeyboardHook


class Win32HookAdapter(HotkeyListener):
    """Global hotkey listener backed by a native Win32 keyboard hook.

    Translates each raw virtual-key event into a :class:`HotkeyKey` and
    delegates to the strategy. It never suppresses keys - a global hotkey
    must not swallow the user's keystrokes.
    """

    def __init__(self, strategy: HotkeyStrategy) -> None:
        self._strategy = strategy
        self._hook = Win32KeyboardHook(self._handle_key)
        self._on_activate: Callable[[], None] | None = None
        self._on_deactivate: Callable[[], None] | None = None
        self._paused: bool = False

    def start(self) -> None:
        self._hook.start()

    def stop(self) -> None:
        self._hook.stop()

    def set_on_activate(self, callback: Callable[[], None]) -> None:
        self._on_activate = callback
        self._update_strategy_callbacks()

    def set_on_deactivate(self, callback: Callable[[], None]) -> None:
        self._on_deactivate = callback
        self._update_strategy_callbacks()

    def _update_strategy_callbacks(self) -> None:
        if self._on_activate is not None and self._on_deactivate is not None:
            self._strategy.set_callbacks(self._on_activate, self._on_deactivate)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _handle_key(self, is_key_down: bool, vk: int) -> bool:
        if self._paused:
            return False
        key = keys.from_vk(vk)
        if is_key_down:
            self._strategy.on_key_press(key)
        else:
            self._strategy.on_key_release(key)
        return False
