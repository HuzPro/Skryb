from voicetotext.hotkeys.keys import (
    VK_F1,
    VK_LCONTROL,
    VK_LMENU,
    VK_LSHIFT,
    VK_LWIN,
    VK_RCONTROL,
    VK_RMENU,
    VK_RSHIFT,
    VK_SPACE,
    HotkeyKey,
    from_char,
)


# A generic modifier name resolves to the LEFT virtual key; the explicit
# ``_l``/``_r`` forms let a config pin a specific side. Both display as the
# same generic name (see ``keys.name_for_vk``).
_MODIFIER_MAP: dict[str, HotkeyKey] = {
    "ctrl": HotkeyKey(VK_LCONTROL, "ctrl"),
    "ctrl_l": HotkeyKey(VK_LCONTROL, "ctrl"),
    "ctrl_r": HotkeyKey(VK_RCONTROL, "ctrl"),
    "alt": HotkeyKey(VK_LMENU, "alt"),
    "alt_l": HotkeyKey(VK_LMENU, "alt"),
    "alt_r": HotkeyKey(VK_RMENU, "alt"),
    "shift": HotkeyKey(VK_LSHIFT, "shift"),
    "shift_l": HotkeyKey(VK_LSHIFT, "shift"),
    "shift_r": HotkeyKey(VK_RSHIFT, "shift"),
    "cmd": HotkeyKey(VK_LWIN, "cmd"),
    "space": HotkeyKey(VK_SPACE, "space"),
    **{f"f{n}": HotkeyKey(VK_F1 + n - 1, f"f{n}") for n in range(1, 13)},
}


def parse_hotkey_combo(combo_string: str) -> tuple[HotkeyKey, ...]:
    parsed: list[HotkeyKey] = []
    for raw_part in combo_string.lower().split("+"):
        part = raw_part.strip()
        if part in _MODIFIER_MAP:
            parsed.append(_MODIFIER_MAP[part])
        elif len(part) == 1 and part.isalpha():
            parsed.append(from_char(part))
        else:
            raise ValueError(f"Unknown key: {part}")
    return tuple(parsed)
