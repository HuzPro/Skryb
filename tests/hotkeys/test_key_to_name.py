"""Tests for ``key_to_canonical_name`` - converting a hook key event back
to the canonical name string used in config (inverse of the parser)."""

from voicetotext.hotkeys import keys
from voicetotext.hotkeys.keys import HotkeyKey
from voicetotext.hotkeys.key_to_name import key_to_canonical_name


class TestModifiers:
    def test_should_convert_left_ctrl_to_ctrl(self):
        assert key_to_canonical_name(HotkeyKey(keys.VK_LCONTROL)) == "ctrl"

    def test_should_convert_right_ctrl_to_ctrl(self):
        assert key_to_canonical_name(HotkeyKey(keys.VK_RCONTROL)) == "ctrl"

    def test_should_convert_left_and_right_alt_to_alt(self):
        assert key_to_canonical_name(HotkeyKey(keys.VK_LMENU)) == "alt"
        assert key_to_canonical_name(HotkeyKey(keys.VK_RMENU)) == "alt"

    def test_should_convert_left_and_right_shift_to_shift(self):
        assert key_to_canonical_name(HotkeyKey(keys.VK_LSHIFT)) == "shift"
        assert key_to_canonical_name(HotkeyKey(keys.VK_RSHIFT)) == "shift"

    def test_should_convert_win_to_cmd(self):
        assert key_to_canonical_name(HotkeyKey(keys.VK_LWIN)) == "cmd"


class TestNamedKeys:
    def test_should_convert_space_to_space(self):
        assert key_to_canonical_name(HotkeyKey(keys.VK_SPACE)) == "space"

    def test_should_convert_f1_and_f12(self):
        assert key_to_canonical_name(HotkeyKey(0x70)) == "f1"
        assert key_to_canonical_name(HotkeyKey(0x7B)) == "f12"

    def test_should_convert_escape_to_esc(self):
        assert key_to_canonical_name(HotkeyKey(keys.VK_ESCAPE)) == "esc"


class TestLettersAndDigits:
    def test_should_convert_letter_to_lowercase_name(self):
        assert key_to_canonical_name(HotkeyKey(0x41)) == "a"
        assert key_to_canonical_name(keys.from_char("M")) == "m"

    def test_should_convert_digit_to_digit_name(self):
        assert key_to_canonical_name(HotkeyKey(0x31)) == "1"


class TestUnknown:
    def test_should_return_none_for_unrecognized_key(self):
        assert key_to_canonical_name(HotkeyKey(0x2C)) is None  # Print Screen

    def test_should_return_none_outside_alphanumeric_range(self):
        assert key_to_canonical_name(HotkeyKey(0x100)) is None
