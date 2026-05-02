from pynput.keyboard import Key, KeyCode

from voicetotext.hotkeys.key_parser import canonicalize_key


_KEY_TO_NAME: dict[Key, str] = {
    Key.ctrl_l: "ctrl",
    Key.ctrl_r: "ctrl",
    Key.alt_l: "alt",
    Key.alt_r: "alt",
    Key.shift_l: "shift",
    Key.shift_r: "shift",
    Key.cmd: "cmd",
    Key.space: "space",
    Key.esc: "esc",
    Key.f1: "f1",
    Key.f2: "f2",
    Key.f3: "f3",
    Key.f4: "f4",
    Key.f5: "f5",
    Key.f6: "f6",
    Key.f7: "f7",
    Key.f8: "f8",
    Key.f9: "f9",
    Key.f10: "f10",
    Key.f11: "f11",
    Key.f12: "f12",
}

_VK_A = 0x41
_VK_Z = 0x5A
_VK_0 = 0x30
_VK_9 = 0x39


def key_to_canonical_name(key: Key | KeyCode) -> str | None:
    if isinstance(key, Key):
        return _KEY_TO_NAME.get(key)
    canonical = canonicalize_key(key)
    if not isinstance(canonical, KeyCode) or canonical.vk is None:
        return None
    vk = canonical.vk
    if _VK_A <= vk <= _VK_Z:
        return chr(vk).lower()
    if _VK_0 <= vk <= _VK_9:
        return chr(vk)
    return None
