import ctypes
import ctypes.wintypes as _wt

from voicetotext.output.modifier_key_state_reader import ModifierKeyStateReader


_user32 = ctypes.windll.user32

_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short

# Generic VK_SHIFT / VK_CONTROL / VK_MENU cover both left and right
# variants in GetAsyncKeyState; the Win keys only exist as sided codes.
_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_MENU = 0x12
_VK_LWIN = 0x5B
_VK_RWIN = 0x5C

_MODIFIER_VIRTUAL_KEYS = (_VK_SHIFT, _VK_CONTROL, _VK_MENU, _VK_LWIN, _VK_RWIN)

# High bit of the SHORT means "currently down"; bit 0 means "pressed
# since the last call" and must be ignored.
_KEY_DOWN_FLAG = 0x8000


class Win32ModifierKeyStateReader(ModifierKeyStateReader):

    def any_modifier_down(self) -> bool:
        return any(
            _user32.GetAsyncKeyState(key) & _KEY_DOWN_FLAG
            for key in _MODIFIER_VIRTUAL_KEYS
        )
