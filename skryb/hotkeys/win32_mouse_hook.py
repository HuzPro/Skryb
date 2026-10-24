import ctypes
import ctypes.wintypes as _wt

from skryb.hotkeys.keys import VK_MBUTTON, VK_XBUTTON1, VK_XBUTTON2
from skryb.hotkeys.win32_keyboard_hook import Win32LowLevelHook

_WH_MOUSE_LL = 14
_WM_MBUTTONDOWN = 0x0207
_WM_MBUTTONUP = 0x0208
_WM_XBUTTONDOWN = 0x020B
_WM_XBUTTONUP = 0x020C
_XBUTTON1 = 0x0001


class _MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", _wt.POINT),
        ("mouseData", _wt.DWORD),
        ("flags", _wt.DWORD),
        ("time", _wt.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class Win32MouseHook(Win32LowLevelHook):
    """Mouse buttons 3-5 as key events for a keyboard-style handler.

    The middle button arrives as ``VK_MBUTTON`` and the side buttons as
    ``VK_XBUTTON1``/``VK_XBUTTON2``, so a strategy matches them like keys.
    This proc also runs for every cursor move, which is why everything
    other than those three buttons is passed on without a look.
    """

    _hook_type = _WH_MOUSE_LL
    _device = "mouse"

    def _dispatch(self, message: int, l_param: int) -> bool:
        if message in (_WM_MBUTTONDOWN, _WM_MBUTTONUP):
            return self._handler(message == _WM_MBUTTONDOWN, VK_MBUTTON)
        if message in (_WM_XBUTTONDOWN, _WM_XBUTTONUP):
            info = ctypes.cast(l_param, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents
            side = VK_XBUTTON1 if info.mouseData >> 16 == _XBUTTON1 else VK_XBUTTON2
            return self._handler(message == _WM_XBUTTONDOWN, side)
        return False
