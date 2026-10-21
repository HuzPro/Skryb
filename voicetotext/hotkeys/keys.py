from dataclasses import dataclass, field


# Win32 virtual-key codes. The low-level keyboard hook reports the distinct
# left/right codes for the modifier keys; we keep them distinct (so an
# explicitly side-specific hotkey still works) while mapping a *generic*
# modifier name to the LEFT code (a generic "ctrl" hotkey fires on Left Ctrl).
VK_LSHIFT = 0xA0
VK_RSHIFT = 0xA1
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3
VK_LMENU = 0xA4  # left Alt
VK_RMENU = 0xA5  # right Alt
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_SPACE = 0x20
VK_ESCAPE = 0x1B
VK_F1 = 0x70  # F1..F24 are contiguous: 0x70..0x87
# Mouse buttons 3-5 have virtual-key codes too; the mouse hook reports them
# under these so one shortcut can mix keys and a button.
VK_MBUTTON = 0x04
VK_XBUTTON1 = 0x05
VK_XBUTTON2 = 0x06
MOUSE_BUTTON_VKS = frozenset({VK_MBUTTON, VK_XBUTTON1, VK_XBUTTON2})

_VK_A = 0x41
_VK_Z = 0x5A
_VK_0 = 0x30
_VK_9 = 0x39

# Windows' VK_OEM_* punctuation keys, named by what they print on a US
# layout. Another layout prints something else on the same physical key.
PUNCTUATION_KEYS: dict[int, str] = {
    0xBA: "semicolon",
    0xBB: "equals",
    0xBC: "comma",
    0xBD: "minus",
    0xBE: "period",
    0xBF: "slash",
    0xC0: "backtick",
    0xDB: "bracket_left",
    0xDC: "backslash",
    0xDD: "bracket_right",
    0xDE: "quote",
}


@dataclass(frozen=True)
class HotkeyKey:
    """A single key identified by its virtual-key code.

    Equality and hashing use ``vk`` ALONE - ``name`` is a display label
    excluded from comparison - so a key delivered by the OS hook compares
    equal to a parsed target regardless of how the target was labelled.
    """

    vk: int
    name: str | None = field(default=None, compare=False)


def from_vk(vk: int) -> HotkeyKey:
    return HotkeyKey(vk)


def from_char(char: str) -> HotkeyKey:
    # Letters ``a``-``z`` and digits ``0``-``9`` share their ASCII value with
    # their virtual-key code; uppercasing folds ``a`` and ``A`` together.
    return HotkeyKey(ord(char.upper()))


# A generic modifier name resolves to the LEFT vk; both side vks fold back
# to that one name for display and capture.
_GENERIC_MODIFIER_NAMES: dict[int, str] = {
    VK_LCONTROL: "ctrl", VK_RCONTROL: "ctrl",
    VK_LMENU: "alt", VK_RMENU: "alt",
    VK_LSHIFT: "shift", VK_RSHIFT: "shift",
    VK_LWIN: "cmd", VK_RWIN: "cmd",
}
MODIFIER_VKS = frozenset(_GENERIC_MODIFIER_NAMES)

_NAMED_KEYS: dict[int, str] = {
    **_GENERIC_MODIFIER_NAMES,
    VK_SPACE: "space",
    VK_ESCAPE: "esc",
    VK_MBUTTON: "mouse_middle",
    VK_XBUTTON1: "mouse4",
    VK_XBUTTON2: "mouse5",
    **{VK_F1 + offset: f"f{offset + 1}" for offset in range(24)},
    **PUNCTUATION_KEYS,
}


def name_for_vk(vk: int) -> str | None:
    """The canonical config name for *vk*, or ``None`` if unsupported."""
    if vk in _NAMED_KEYS:
        return _NAMED_KEYS[vk]
    if _VK_A <= vk <= _VK_Z:
        return chr(vk).lower()
    if _VK_0 <= vk <= _VK_9:
        return chr(vk)
    return None
