from skryb.hotkeys.keys import HotkeyKey, name_for_vk


def key_to_canonical_name(key: HotkeyKey) -> str | None:
    """The canonical config name for *key*, or ``None`` if unsupported.

    Inverse of ``parse_hotkey_combo``: both left and right modifier vks
    fold to one generic name, letters lowercase, digits pass through.
    """
    return name_for_vk(key.vk)
