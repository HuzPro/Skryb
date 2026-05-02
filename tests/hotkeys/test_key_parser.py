"""Tests for ``parse_hotkey_combo`` and ``canonicalize_key``.

Parsing is pure — no OS or pynput runtime state — so tests drive it
through the public function and assert on the resulting key tuple.
"""

import pytest
from pynput.keyboard import Key, KeyCode

from voicetotext.hotkeys.key_parser import canonicalize_key, parse_hotkey_combo


class TestParseHotkeyCombo:
    def test_should_return_tuple(self):
        keys = parse_hotkey_combo("ctrl+alt+m")

        assert isinstance(keys, tuple)

    def test_should_parse_single_modifier(self):
        keys = parse_hotkey_combo("ctrl")

        assert keys == (Key.ctrl_l,)

    def test_should_parse_single_letter_key(self):
        keys = parse_hotkey_combo("a")

        assert len(keys) == 1
        assert isinstance(keys[0], KeyCode)
        assert keys[0].vk == 0x41

    def test_should_parse_single_function_key(self):
        keys = parse_hotkey_combo("f1")

        assert keys == (Key.f1,)

    def test_should_preserve_declared_key_order(self):
        keys = parse_hotkey_combo("alt+ctrl+m")

        assert keys[0] == Key.alt_l
        assert keys[1] == Key.ctrl_l
        assert isinstance(keys[2], KeyCode) and keys[2].vk == 0x4D

    def test_should_parse_ctrl_alt_letter_combo(self):
        keys = parse_hotkey_combo("ctrl+alt+m")

        assert keys[0] == Key.ctrl_l
        assert keys[1] == Key.alt_l
        assert isinstance(keys[2], KeyCode) and keys[2].vk == 0x4D

    def test_should_distinguish_different_orderings(self):
        a = parse_hotkey_combo("ctrl+alt+m")
        b = parse_hotkey_combo("alt+m+ctrl")

        assert a != b

    def test_should_be_case_insensitive(self):
        lower = parse_hotkey_combo("ctrl+alt+m")
        upper = parse_hotkey_combo("CTRL+ALT+M")

        assert lower == upper

    def test_should_ignore_surrounding_whitespace_around_parts(self):
        keys = parse_hotkey_combo("ctrl + alt + m")

        assert keys[0] == Key.ctrl_l
        assert keys[1] == Key.alt_l

    def test_should_parse_ctrl_shift_alt_combo(self):
        keys = parse_hotkey_combo("ctrl+shift+alt+x")

        assert keys[0] == Key.ctrl_l
        assert keys[1] == Key.shift_l
        assert keys[2] == Key.alt_l

    def test_should_raise_for_unknown_multi_letter_name(self):
        with pytest.raises(ValueError, match="Unknown key"):
            parse_hotkey_combo("nonsense")

    def test_should_distinguish_left_and_right_modifiers_when_explicit(self):
        left = parse_hotkey_combo("ctrl_l")
        right = parse_hotkey_combo("ctrl_r")

        assert left == (Key.ctrl_l,)
        assert right == (Key.ctrl_r,)


class TestCanonicalizeKey:
    def test_should_leave_non_keycode_instances_untouched(self):
        result = canonicalize_key(Key.ctrl_l)

        assert result is Key.ctrl_l

    def test_should_normalise_letter_keycode_to_virtual_key(self):
        raw = KeyCode.from_char("a")

        canonical = canonicalize_key(raw)

        assert isinstance(canonical, KeyCode)
        assert canonical.vk == 0x41

    def test_should_normalise_uppercase_letter_keycode(self):
        raw = KeyCode.from_char("A")

        canonical = canonicalize_key(raw)

        assert canonical.vk == 0x41
