import threading
from collections.abc import Callable, Sequence

from voicetotext.hotkeys import keys
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.strategy import HotkeyStrategy
from voicetotext.hotkeys.win32_keyboard_hook import (
    KeyObserver,
    Win32KeyboardHook,
    ignore_key_event,
)
from voicetotext.hotkeys.win32_mouse_hook import Win32MouseHook


def _never_cancels() -> bool:
    return False


class Win32HookAdapter(HotkeyListener):
    """Global shortcut listener backed by native Win32 input hooks.

    Offers every key and mouse-button event to each shortcut's strategy.
    Keys always reach the focused application, the shortcuts' own included;
    the one exception is Esc, and only when ``cancel`` says it cancelled a
    recording with it. A mouse button is swallowed, down and up, when it
    completes a shortcut, so a side button bound to dictation does not
    also send the browser back a page.

    The mouse hook is installed only when a shortcut uses a button: its
    proc runs on every cursor move. The two hooks run on two threads, so
    the strategies are only ever touched under one lock.
    """

    def __init__(
        self,
        strategies: Sequence[HotkeyStrategy],
        observer: KeyObserver = ignore_key_event,
        cancel: Callable[[], bool] = _never_cancels,
        *,
        listen_to_mouse: bool = False,
    ) -> None:
        self._strategies = tuple(strategies)
        self._cancel = cancel
        self._swallowing_escape = False
        self._swallowed_buttons: set[int] = set()
        self._lock = threading.Lock()
        self._hook = Win32KeyboardHook(self._handle_key, observer)
        self._mouse_hook = Win32MouseHook(self._handle_key) if listen_to_mouse else None
        self._paused: bool = False

    def start(self) -> None:
        self._hook.start()
        if self._mouse_hook is not None:
            self._mouse_hook.start()

    def stop(self) -> None:
        try:
            self._hook.stop()
        finally:
            if self._mouse_hook is not None:
                self._mouse_hook.stop()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _handle_key(self, is_key_down: bool, vk: int) -> bool:
        if self._paused:
            return False
        with self._lock:
            if vk == keys.VK_ESCAPE and self._handle_escape(is_key_down):
                return True
            key = keys.from_vk(vk)
            if is_key_down:
                completed = [strategy.on_key_press(key) for strategy in self._strategies]
                return self._swallow_button_down(vk, any(completed))
            for strategy in self._strategies:
                strategy.on_key_release(key)
            return self._swallow_button_up(vk)

    def _swallow_button_down(self, vk: int, completed_a_shortcut: bool) -> bool:
        if vk not in keys.MOUSE_BUTTON_VKS or not completed_a_shortcut:
            return False
        self._swallowed_buttons.add(vk)
        return True

    def _swallow_button_up(self, vk: int) -> bool:
        if vk not in self._swallowed_buttons:
            return False
        self._swallowed_buttons.discard(vk)
        return True

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
