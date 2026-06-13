import ctypes
import ctypes.wintypes as _wt

from voicetotext.core.exceptions import OutputError


_user32 = ctypes.windll.user32

_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_VK_CONTROL = 0x11
_VK_V = 0x56

# ``dwExtraInfo`` is an ULONG_PTR — it MUST be pointer-sized or SendInput
# silently injects nothing on 64-bit Windows. ``c_size_t`` matches a
# pointer's width on both 32- and 64-bit builds.
_ULONG_PTR = ctypes.c_size_t


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", _wt.LONG),
        ("dy", _wt.LONG),
        ("mouseData", _wt.DWORD),
        ("dwFlags", _wt.DWORD),
        ("time", _wt.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _KeybdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", _wt.WORD),
        ("wScan", _wt.WORD),
        ("dwFlags", _wt.DWORD),
        ("time", _wt.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    ]


class _HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", _wt.DWORD),
        ("wParamL", _wt.WORD),
        ("wParamH", _wt.WORD),
    ]


# The union must carry every member so ``sizeof(_Input)`` matches the real
# Win32 ``INPUT`` (sized to the largest, MOUSEINPUT). SendInput rejects the
# batch when the caller-supplied struct size is wrong.
class _InputUnion(ctypes.Union):
    _fields_ = [
        ("mi", _MouseInput),
        ("ki", _KeybdInput),
        ("hi", _HardwareInput),
    ]


class _Input(ctypes.Structure):
    _fields_ = [
        ("type", _wt.DWORD),
        ("union", _InputUnion),
    ]


_user32.SendInput.argtypes = [_wt.UINT, ctypes.POINTER(_Input), ctypes.c_int]
_user32.SendInput.restype = _wt.UINT


class Win32KeySender:

    def send_ctrl_v(self) -> None:
        _send_ctrl_v()


def _key_event(virtual_key: int, *, key_up: bool) -> _Input:
    event = _Input()
    event.type = _INPUT_KEYBOARD
    event.union.ki = _KeybdInput(
        wVk=virtual_key,
        wScan=0,
        dwFlags=_KEYEVENTF_KEYUP if key_up else 0,
        time=0,
        dwExtraInfo=0,
    )
    return event


def _send_ctrl_v() -> None:
    events = (_Input * 4)(
        _key_event(_VK_CONTROL, key_up=False),
        _key_event(_VK_V, key_up=False),
        _key_event(_VK_V, key_up=True),
        _key_event(_VK_CONTROL, key_up=True),
    )
    sent = _user32.SendInput(len(events), events, ctypes.sizeof(_Input))
    if sent != len(events):
        raise OutputError(
            f"SendInput injected {sent} of {len(events)} Ctrl+V key events."
        )
