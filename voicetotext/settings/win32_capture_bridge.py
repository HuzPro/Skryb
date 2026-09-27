from voicetotext.hotkeys import keys
from voicetotext.hotkeys.win32_keyboard_hook import Win32KeyboardHook
from voicetotext.hotkeys.win32_mouse_hook import Win32MouseHook
from voicetotext.settings.signal import Signal


class Win32CaptureBridge:
    """Suppressing key-capture bridge used while recording a shortcut.

    Installs native low-level keyboard and mouse hooks that swallow every
    key and mouse buttons 3-5 (so OS shortcuts and a browser's back button
    cannot fire mid-recording; left and right clicks still reach the Stop
    button), and emits the canonical name of each one pressed through a
    callback-list :class:`Signal`. The emission runs on a hook's worker
    thread; the consuming widget marshals it onto the GUI thread.

    Several capture widgets share one bridge, so the hooks stay installed
    until every ``start`` has been matched by a ``stop``.
    """

    def __init__(self) -> None:
        self.key_pressed = Signal()
        self._hook = Win32KeyboardHook(self._handle_key)
        self._mouse_hook = Win32MouseHook(self._handle_key)
        self._users = 0

    def start(self) -> None:
        self._users += 1
        if self._users > 1:
            return
        self._hook.start()
        self._mouse_hook.start()

    def stop(self) -> None:
        if self._users == 0:
            return
        self._users -= 1
        if self._users:
            return
        # A keyboard hook stuck on its thread must not leave the mouse hook
        # swallowing the side buttons for good.
        try:
            self._hook.stop()
        finally:
            self._mouse_hook.stop()

    def _handle_key(self, is_key_down: bool, vk: int) -> bool:
        if is_key_down:
            name = keys.name_for_vk(vk)
            if name is not None:
                self.key_pressed.emit(name)
        return True  # suppress every key while recording
