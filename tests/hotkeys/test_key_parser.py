"""Tests for ``parse_hotkey_combo``.

Parsing is pure - no OS state - so tests drive it through the public
function and assert on the resulting ``HotkeyKey`` tuple (compared by vk).
"""

import pytest

from skryb.hotkeys import keys
from skryb.hotkeys.keys import HotkeyKey
from skryb.hotkeys.key_parser import parse_hotkey_combo


class TestParseHotkeyCombo:
    def test_should_return_tuple(self):
        assert isinstance(parse_hotkey_combo("ctrl+alt+m"), tuple)

    def test_should_parse_single_modifier_to_left_vk(self):
        assert parse_hotkey_combo("ctrl") == (HotkeyKey(keys.VK_LCONTROL),)

    def test_should_parse_single_letter_key(self):
        parsed = parse_hotkey_combo("a")

        assert len(parsed) == 1
        assert parsed[0].vk == 0x41

    def test_should_parse_single_function_key(self):
        assert parse_hotkey_combo("f1") == (HotkeyKey(0x70),)

    def test_should_preserve_declared_key_order(self):
        parsed = parse_hotkey_combo("alt+ctrl+m")

        assert parsed[0].vk == keys.VK_LMENU
        assert parsed[1].vk == keys.VK_LCONTROL
        assert parsed[2].vk == 0x4D

    def test_should_parse_ctrl_alt_letter_combo(self):
        parsed = parse_hotkey_combo("ctrl+alt+m")

        assert parsed[0].vk == keys.VK_LCONTROL
        assert parsed[1].vk == keys.VK_LMENU
        assert parsed[2].vk == 0x4D

    def test_should_distinguish_different_orderings(self):
        assert parse_hotkey_combo("ctrl+alt+m") != parse_hotkey_combo("alt+m+ctrl")

    def test_should_be_case_insensitive(self):
        assert parse_hotkey_combo("ctrl+alt+m") == parse_hotkey_combo("CTRL+ALT+M")

    def test_should_ignore_surrounding_whitespace_around_parts(self):
        parsed = parse_hotkey_combo("ctrl + alt + m")

        assert parsed[0].vk == keys.VK_LCONTROL
        assert parsed[1].vk == keys.VK_LMENU

    def test_should_parse_ctrl_shift_alt_combo(self):
        parsed = parse_hotkey_combo("ctrl+shift+alt+x")

        assert parsed[0].vk == keys.VK_LCONTROL
        assert parsed[1].vk == keys.VK_LSHIFT
        assert parsed[2].vk == keys.VK_LMENU

    def test_should_raise_for_unknown_multi_letter_name(self):
        with pytest.raises(ValueError, match="Unknown key"):
            parse_hotkey_combo("nonsense")

    def test_should_distinguish_left_and_right_modifiers_when_explicit(self):
        left = parse_hotkey_combo("ctrl_l")
        right = parse_hotkey_combo("ctrl_r")

        assert left[0].vk == keys.VK_LCONTROL
        assert right[0].vk == keys.VK_RCONTROL
        assert left != right
