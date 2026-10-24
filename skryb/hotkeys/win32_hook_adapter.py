import logging
import threading
from collections.abc import Callable, Sequence

from skryb.core.exceptions import OutputError
from skryb.hotkeys import keys
from skryb.hotkeys.listener import HotkeyListener
from skryb.hotkeys.strategy import HotkeyStrategy
from skryb.hotkeys.win32_keyboard_hook import (
    KeyObserver,
    Win32KeyboardHook,
    ignore_key_event,
)
from skryb.hotkeys.win32_mouse_hook import Win32MouseHook

_logger = logging.getLogger(__name__)

# Released on their own, these open the Start menu or the focused app's menu bar.
_MENU_KEYS = frozenset({keys.VK_LMENU, keys.VK_RMENU, keys.VK_LWIN, keys.VK_RWIN})


def _never_cancels() -> bool:
    return False


def _no_mask() -> None:
    return None


class Win32HookAdapter(HotkeyListener):
    """Global shortcut listener backed by native Win32 input hooks.

    Offers every key and mouse-button event to each shortcut's strategy.
    The key or button that completes a shortcut is swallowed, down, repeats
    and up, so the focused app never sees it: Ctrl+Alt+M reaches the app as
    a bare Ctrl+Alt, and a side button bound to dictation does not also send
    the browser back a page. Every other key reaches the app, the
    shortcut's modifiers included. Esc is swallowed too, but only when
    ``cancel`` says it cancelled a recording with it.

    Swallowing the key between a Win or Alt press and its release makes that
    release look like a lone tap, which opens the Start menu or a menu bar,
    so ``mask_menus`` then taps a key that does nothing.

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
        mask_menus: Callable[[], None] = _no_mask,
    ) -> None:
        self._strategies = tuple(strategies)
        self._cancel = cancel
        self._mask_menus = mask_menus
        self._swallowing_escape = False
        self._swallowed: set[int] = set()
        self._held_menu_keys: set[int] = set()
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
        # A key released while paused never reports its release here, and
        # one left behind would swallow its next press. A stale held Win or
        # Alt only sends a needless mask key, so those are kept.
        with self._lock:
            self._swallowed.clear()

    def resume(self) -> None:
        self._paused = False

    def _handle_key(self, is_key_down: bool, vk: int) -> bool:
        if self._paused:
            return False
        with self._lock:
            self._track_menu_key(is_key_down, vk)
            if vk == keys.VK_ESCAPE and self._handle_escape(is_key_down):
                return True
            key = keys.from_vk(vk)
            if is_key_down:
                completed = [strategy.on_key_press(key) for strategy in self._strategies]
                return self._swallow_down(vk, any(completed))
            for strategy in self._strategies:
                strategy.on_key_release(key)
            return self._swallow_up(vk)

    def _track_menu_key(self, is_key_down: bool, vk: int) -> None:
        if vk not in _MENU_KEYS:
            return
        if is_key_down:
            self._held_menu_keys.add(vk)
        else:
            self._held_menu_keys.discard(vk)

    def _swallow_down(self, vk: int, completed_a_shortcut: bool) -> bool:
        if vk in self._swallowed:
            return True  # a held key repeating
        if not completed_a_shortcut:
            return False
        self._swallowed.add(vk)
        if self._held_menu_keys:
            self._keep_menus_closed()
        return True

    def _swallow_up(self, vk: int) -> bool:
        if vk not in self._swallowed:
            return False
        self._swallowed.discard(vk)
        return True

    def _keep_menus_closed(self) -> None:
        try:
            self._mask_menus()
        except OutputError:
            # An elevated window in front refuses injected keys.
            _logger.debug("The menu mask key was refused.", exc_info=True)

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
