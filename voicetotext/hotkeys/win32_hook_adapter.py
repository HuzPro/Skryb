from collections.abc import Callable

from voicetotext.hotkeys import keys
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.strategy import HotkeyStrategy
from voicetotext.hotkeys.win32_keyboard_hook import (
    KeyObserver,
    Win32KeyboardHook,
    ignore_key_event,
)


def _never_cancels() -> bool:
    return False


class Win32HookAdapter(HotkeyListener):
    """Global hotkey listener backed by a native Win32 keyboard hook.

    Translates each raw virtual-key event into a :class:`HotkeyKey` and
    delegates to the strategy. It swallows only one key: Esc, and only when
    ``cancel`` says it cancelled a recording with it. Every other key, the
    hotkey included, reaches the focused application untouched.
    """

    def __init__(
        self,
        strategy: HotkeyStrategy,
        observer: KeyObserver = ignore_key_event,
        cancel: Callable[[], bool] = _never_cancels,
    ) -> None:
        self._strategy = strategy
        self._cancel = cancel
        self._swallowing_escape = False
        self._hook = Win32KeyboardHook(self._handle_key, observer)
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
        if vk == keys.VK_ESCAPE and self._handle_escape(is_key_down):
            return True
        key = keys.from_vk(vk)
        if is_key_down:
            self._strategy.on_key_press(key)
        else:
            self._strategy.on_key_release(key)
        return False

    def _handle_escape(self, is_key_down: bool) -> bool:
        # The release of a swallowed Esc is swallowed too, so the app never
        # sees half a keystroke.
        if not is_key_down:
            swallowed = self._swallowing_escape
            self._swallowing_escape = False
            return swallowed
        if self._swallowing_escape or self._cancel():
            self._swallowing_escape = True
            return True
        return False
