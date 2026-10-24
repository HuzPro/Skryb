from skryb.hotkeys import keys
from skryb.hotkeys.keys import HotkeyKey


class TestHotkeyKeyEquality:
    def test_should_be_equal_when_vk_matches_regardless_of_name(self):
        assert HotkeyKey(0x11, "ctrl") == HotkeyKey(0x11, "something else")

    def test_should_not_be_equal_when_vk_differs(self):
        assert HotkeyKey(0x11) != HotkeyKey(0x12)

    def test_should_hash_by_vk_only(self):
        assert hash(HotkeyKey(0x4D, "m")) == hash(HotkeyKey(0x4D, None))

    def test_should_be_usable_as_a_set_member_by_vk(self):
        held = {HotkeyKey(0x4D, "m")}

        assert HotkeyKey(0x4D, "ignored") in held


class TestFromVk:
    def test_should_carry_the_given_vk(self):
        assert keys.from_vk(0xA2).vk == 0xA2


class TestFromChar:
    def test_should_map_lowercase_letter_to_its_virtual_key(self):
        assert keys.from_char("a").vk == 0x41

    def test_should_map_uppercase_letter_to_the_same_virtual_key(self):
        assert keys.from_char("A").vk == keys.from_char("a").vk

    def test_should_map_letter_m_to_0x4d(self):
        assert keys.from_char("m").vk == 0x4D

    def test_should_map_digit_to_its_virtual_key(self):
        assert keys.from_char("1").vk == 0x31


class TestNameForVk:
    def test_should_name_left_control_as_ctrl(self):
        assert keys.name_for_vk(keys.VK_LCONTROL) == "ctrl"

    def test_should_name_right_control_as_ctrl(self):
        assert keys.name_for_vk(keys.VK_RCONTROL) == "ctrl"

    def test_should_name_left_and_right_alt_as_alt(self):
        assert keys.name_for_vk(keys.VK_LMENU) == "alt"
        assert keys.name_for_vk(keys.VK_RMENU) == "alt"

    def test_should_name_left_and_right_shift_as_shift(self):
        assert keys.name_for_vk(keys.VK_LSHIFT) == "shift"
        assert keys.name_for_vk(keys.VK_RSHIFT) == "shift"

    def test_should_name_win_key_as_cmd(self):
        assert keys.name_for_vk(keys.VK_LWIN) == "cmd"

    def test_should_name_space_and_escape(self):
        assert keys.name_for_vk(keys.VK_SPACE) == "space"
        assert keys.name_for_vk(keys.VK_ESCAPE) == "esc"

    def test_should_name_function_keys(self):
        assert keys.name_for_vk(0x70) == "f1"
        assert keys.name_for_vk(0x7B) == "f12"

    def test_should_name_letters_lowercase(self):
        assert keys.name_for_vk(0x41) == "a"
        assert keys.name_for_vk(0x4D) == "m"

    def test_should_name_digits(self):
        assert keys.name_for_vk(0x31) == "1"

    def test_should_return_none_for_unrecognized_key(self):
        assert keys.name_for_vk(0x2C) is None  # Print Screen

    def test_should_return_none_outside_alphanumeric_range(self):
        assert keys.name_for_vk(0x100) is None
