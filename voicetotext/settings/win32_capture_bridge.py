from voicetotext.hotkeys import keys
from voicetotext.hotkeys.win32_keyboard_hook import Win32KeyboardHook
from voicetotext.settings.signal import Signal


class Win32CaptureBridge:
    """Suppressing key-capture bridge used while recording a hotkey.

    Installs a native low-level keyboard hook that swallows every key (so
    OS shortcuts cannot fire mid-recording) and emits the canonical name of
    each pressed key through a callback-list :class:`Signal`. The emission
    runs on the hook's worker thread; the consuming widget marshals it onto
    the GUI thread.
    """

    def __init__(self) -> None:
        self.key_pressed = Signal()
        self._hook = Win32KeyboardHook(self._handle_key)

    def start(self) -> None:
        self._hook.start()

    def stop(self) -> None:
        self._hook.stop()

    def _handle_key(self, is_key_down: bool, vk: int) -> bool:
        if is_key_down:
            name = keys.name_for_vk(vk)
            if name is not None:
                self.key_pressed.emit(name)
        return True  # suppress every key while recording
