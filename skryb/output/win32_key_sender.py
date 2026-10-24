import ctypes
import ctypes.wintypes as _wt

from skryb.core.exceptions import OutputError
from skryb.output.paste_method import PasteMethod, utf16_chunks


_user32 = ctypes.windll.user32

_INPUT_KEYBOARD = 1
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_INSERT = 0x2D
_VK_V = 0x56
# Reserved to mean "no key": apps ignore it, yet it still counts as a key
# pressed between a Win or Alt press and its release.
_VK_NONE = 0xFF

# Insert sits on the navigation block; without the extended flag an app
# reading the scan code sees numpad 0.
_EXTENDED_KEYS = frozenset({_VK_INSERT})

_COMBINATIONS = {
    PasteMethod.CTRL_V: (_VK_CONTROL, _VK_V),
    PasteMethod.CTRL_SHIFT_V: (_VK_CONTROL, _VK_SHIFT, _VK_V),
    PasteMethod.SHIFT_INSERT: (_VK_SHIFT, _VK_INSERT),
}

# ``dwExtraInfo`` is an ULONG_PTR - it MUST be pointer-sized or SendInput
# silently injects nothing on 64-bit Windows. ``c_size_t`` matches a
# pointer's width on both 32- and 64-bit builds.
_ULONG_PTR = ctypes.c_size_t

# A low-level keyboard hook receives ``dwExtraInfo`` verbatim, so stamping
# it identifies our own injected keys exactly. The ``LLKHF_INJECTED`` flag
# would not: it is set for AutoHotkey, PowerToys and the on-screen keyboard
# too. Stays inside 32 bits so it is a valid ULONG_PTR on either build.
SKRYB_INPUT_TAG = 0x56325458


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

    def send_paste(self, method: PasteMethod, text: str) -> None:
        if method is PasteMethod.TYPE:
            _type_text(text)
        else:
            _send_combination(_COMBINATIONS[method])


def send_menu_mask() -> None:
    """Tap a key that does nothing, so releasing a held Win or Alt is not a
    lone tap that opens the Start menu or the focused app's menu bar."""
    _send(
        [_key_event(_VK_NONE, key_up=False), _key_event(_VK_NONE, key_up=True)],
        "menu mask",
    )


def _key_event(virtual_key: int, *, key_up: bool) -> _Input:
    flags = _KEYEVENTF_KEYUP if key_up else 0
    if virtual_key in _EXTENDED_KEYS:
        flags |= _KEYEVENTF_EXTENDEDKEY
    event = _Input()
    event.type = _INPUT_KEYBOARD
    event.union.ki = _KeybdInput(
        wVk=virtual_key,
        wScan=0,
        dwFlags=flags,
        time=0,
        dwExtraInfo=SKRYB_INPUT_TAG,
    )
    return event


def _unicode_event(code_unit: int, *, key_up: bool) -> _Input:
    event = _Input()
    event.type = _INPUT_KEYBOARD
    event.union.ki = _KeybdInput(
        wVk=0,
        wScan=code_unit,
        dwFlags=_KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if key_up else 0),
        time=0,
        dwExtraInfo=SKRYB_INPUT_TAG,
    )
    return event


def _send_combination(virtual_keys: tuple[int, ...]) -> None:
    presses = [_key_event(vk, key_up=False) for vk in virtual_keys]
    releases = [_key_event(vk, key_up=True) for vk in reversed(virtual_keys)]
    _send([*presses, *releases], "paste")


def _type_text(text: str) -> None:
    for chunk in utf16_chunks(text):
        _send(
            [
                event
                for unit in chunk
                for event in (
                    _unicode_event(unit, key_up=False),
                    _unicode_event(unit, key_up=True),
                )
            ],
            "typed",
        )


def _send(inputs: list[_Input], what: str) -> None:
    events = (_Input * len(inputs))(*inputs)
    sent = _user32.SendInput(len(events), events, ctypes.sizeof(_Input))
    if sent != len(events):
        raise OutputError(
            f"SendInput injected {sent} of {len(events)} {what} key events."
        )
