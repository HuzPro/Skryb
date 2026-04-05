from pynput.keyboard import Key, KeyCode

_MODIFIER_MAP = {
    "ctrl": Key.ctrl_l,
    "ctrl_l": Key.ctrl_l,
    "ctrl_r": Key.ctrl_r,
    "alt": Key.alt_l,
    "alt_l": Key.alt_l,
    "alt_r": Key.alt_r,
    "shift": Key.shift_l,
    "shift_l": Key.shift_l,
    "shift_r": Key.shift_r,
    "cmd": Key.cmd,
    "space": Key.space,
    "f1": Key.f1,
    "f2": Key.f2,
    "f3": Key.f3,
    "f4": Key.f4,
    "f5": Key.f5,
    "f6": Key.f6,
    "f7": Key.f7,
    "f8": Key.f8,
    "f9": Key.f9,
    "f10": Key.f10,
    "f11": Key.f11,
    "f12": Key.f12,
}


def parse_hotkey_combo(combo_string: str) -> set:
    keys = set()
    for part in combo_string.lower().split("+"):
        part = part.strip()
        if part in _MODIFIER_MAP:
            keys.add(_MODIFIER_MAP[part])
        elif len(part) == 1:
            keys.add(KeyCode.from_char(part))
        else:
            raise ValueError(f"Unknown key: {part}")
    return keys
