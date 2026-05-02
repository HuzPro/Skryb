"""Tests for the inverse of ``parse_hotkey_combo`` — converting a single
pynput key event back to the canonical name string used in config."""

from pynput.keyboard import Key, KeyCode

from voicetotext.hotkeys.key_to_name import key_to_canonical_name


class TestModifiers:
    def test_should_convert_left_ctrl_to_ctrl(self):
        assert key_to_canonical_name(Key.ctrl_l) == "ctrl"

    def test_should_convert_right_ctrl_to_ctrl(self):
        assert key_to_canonical_name(Key.ctrl_r) == "ctrl"

    def test_should_convert_left_alt_to_alt(self):
        assert key_to_canonical_name(Key.alt_l) == "alt"

    def test_should_convert_right_alt_to_alt(self):
        assert key_to_canonical_name(Key.alt_r) == "alt"

    def test_should_convert_left_shift_to_shift(self):
        assert key_to_canonical_name(Key.shift_l) == "shift"

    def test_should_convert_right_shift_to_shift(self):
        assert key_to_canonical_name(Key.shift_r) == "shift"

    def test_should_convert_cmd_to_cmd(self):
        assert key_to_canonical_name(Key.cmd) == "cmd"


class TestNamedKeys:
    def test_should_convert_space_to_space(self):
        assert key_to_canonical_name(Key.space) == "space"

    def test_should_convert_f1_to_f1(self):
        assert key_to_canonical_name(Key.f1) == "f1"

    def test_should_convert_f12_to_f12(self):
        assert key_to_canonical_name(Key.f12) == "f12"

    def test_should_convert_escape_to_esc(self):
        assert key_to_canonical_name(Key.esc) == "esc"


class TestLetters:
    def test_should_convert_letter_keycode_with_vk_to_letter_name(self):
        assert key_to_canonical_name(KeyCode.from_vk(0x41)) == "a"

    def test_should_convert_letter_keycode_with_char_to_letter_name(self):
        assert key_to_canonical_name(KeyCode.from_char("m")) == "m"

    def test_should_convert_uppercase_letter_to_lowercase(self):
        assert key_to_canonical_name(KeyCode.from_char("M")) == "m"


class TestDigits:
    def test_should_convert_digit_keycode_to_digit_name(self):
        assert key_to_canonical_name(KeyCode.from_vk(0x31)) == "1"


class TestUnknown:
    def test_should_return_none_for_unrecognized_key(self):
        assert key_to_canonical_name(Key.print_screen) is None

    def test_should_return_none_for_keycode_outside_alphanumeric_range(self):
        assert key_to_canonical_name(KeyCode.from_vk(0x100)) is None
