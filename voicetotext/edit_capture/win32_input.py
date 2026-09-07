import ctypes
import ctypes.wintypes as _wt


_user32 = ctypes.windll.user32

_KEY_STATE_SIZE = 256
_KEY_DOWN_FLAG = 0x80
_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_MENU = 0x12
_VK_CAPITAL = 0x14
_TOGGLED_ON = 0x01

_MAPVK_VK_TO_VSC = 0
_TRANSLATION_BUFFER_CHARS = 8
_FIRST_PRINTABLE_CHARACTER = "\x20"

# ``ToUnicodeEx`` normally consumes a pending dead key and rewrites the
# kernel's keyboard state. Bit 0x04 asks what the key would type without
# that side effect, so answering the question cannot corrupt the accented
# character the user is actually in the middle of typing.
_LEAVE_KERNEL_KEY_STATE_ALONE = 0x04

_VK_LBUTTON = 0x01
_VK_RBUTTON = 0x02
_VK_MBUTTON = 0x04
_MOUSE_BUTTONS = (_VK_LBUTTON, _VK_RBUTTON, _VK_MBUTTON)

# Bit 0 of GetAsyncKeyState means "pressed since the last call to this
# function", and reading it clears it. Nothing else in this app polls the
# mouse buttons, so this module is their only reader.
_PRESSED_SINCE_LAST_CALL = 0x0001

_user32.GetForegroundWindow.argtypes = []
_user32.GetForegroundWindow.restype = _wt.HWND
_user32.GetWindowThreadProcessId.argtypes = [
    _wt.HWND, ctypes.POINTER(_wt.DWORD),
]
_user32.GetWindowThreadProcessId.restype = _wt.DWORD
_user32.GetKeyboardLayout.argtypes = [_wt.DWORD]
_user32.GetKeyboardLayout.restype = _wt.HKL
_user32.MapVirtualKeyExW.argtypes = [_wt.UINT, _wt.UINT, _wt.HKL]
_user32.MapVirtualKeyExW.restype = _wt.UINT
_user32.ToUnicodeEx.argtypes = [
    _wt.UINT,
    _wt.UINT,
    ctypes.POINTER(ctypes.c_ubyte),
    _wt.LPWSTR,
    ctypes.c_int,
    _wt.UINT,
    _wt.HKL,
]
_user32.ToUnicodeEx.restype = ctypes.c_int
_user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
_user32.GetAsyncKeyState.restype = ctypes.c_short
_user32.GetKeyState.argtypes = [ctypes.c_int]
_user32.GetKeyState.restype = ctypes.c_short


def caps_lock_enabled() -> bool:
    return bool(_user32.GetKeyState(_VK_CAPITAL) & _TOGGLED_ON)


def key_character(
    vk: int, shift: bool, ctrl: bool, alt: bool, *, caps_lock: bool = False,
) -> str:
    """The character this key would type on the focused window's layout.

    Empty when the key types nothing: a dead key awaiting its base letter,
    a key the layout does not translate, or a control code such as the
    ``\\x01`` Ctrl+A produces.
    """
    layout = _foreground_keyboard_layout()
    buffer = ctypes.create_unicode_buffer(_TRANSLATION_BUFFER_CHARS)
    written = _user32.ToUnicodeEx(
        vk,
        _user32.MapVirtualKeyExW(vk, _MAPVK_VK_TO_VSC, layout),
        _key_state(shift, ctrl, alt, caps_lock=caps_lock),
        buffer,
        _TRANSLATION_BUFFER_CHARS,
        _LEAVE_KERNEL_KEY_STATE_ALONE,
        layout,
    )
    if written <= 0:
        return ""
    typed = "".join(buffer[:written])
    if typed[0] < _FIRST_PRINTABLE_CHARACTER:
        return ""
    return typed


def _key_state(
    shift: bool, ctrl: bool, alt: bool, *, caps_lock: bool = False,
) -> ctypes.Array[ctypes.c_ubyte]:
    # Built from the recorded modifiers rather than read with
    # GetKeyboardState, so a key the user happens to be holding right now
    # cannot change what a past keystroke is said to have typed.
    state = (ctypes.c_ubyte * _KEY_STATE_SIZE)()
    state[_VK_CAPITAL] = _TOGGLED_ON if caps_lock else 0
    for held, modifier in ((shift, _VK_SHIFT), (ctrl, _VK_CONTROL), (alt, _VK_MENU)):
        if held:
            state[modifier] = _KEY_DOWN_FLAG
    return state


def _foreground_keyboard_layout() -> int:
    pid = _wt.DWORD(0)
    thread_id = _user32.GetWindowThreadProcessId(
        _user32.GetForegroundWindow(), ctypes.byref(pid),
    )
    return int(_user32.GetKeyboardLayout(thread_id))


def clicked_since_last_check() -> bool:
    """Whether any mouse button went down since the previous call.

    A click moves the caret without producing a keystroke, so a replay
    driven by keystrokes alone would drift from the real text box with no
    sign anything went wrong. Call this once when a capture block opens to
    discard clicks that predate it.
    """
    # ponytail: blind to a click that no keystroke follows, since only a
    # later call reports it. Upgrade to a WH_MOUSE_LL hook if that proves
    # too coarse.
    #
    # Every button is read on every call - short-circuiting would leave an
    # unread button's bit set and report that stale click next time.
    pressed = [
        _user32.GetAsyncKeyState(button) & _PRESSED_SINCE_LAST_CALL
        for button in _MOUSE_BUTTONS
    ]
    return any(pressed)
