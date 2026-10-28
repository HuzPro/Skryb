import re

import pytest

from tests.fakes.config import FakeConfig
from skryb.hotkeys import keys
from skryb.hotkeys.bindings import (
    DEFAULT_BINDINGS,
    Binding,
    BindingAction,
    BindingError,
    BindingMode,
    bindings_to_config,
    parse_bindings,
    validate_bindings,
)
from skryb.hotkeys.key_parser import parse_hotkey_combo

_TOGGLE = BindingMode.TOGGLE
_HOLD = BindingMode.HOLD
_DICTATE = BindingAction.DICTATE


def _dictate(combo: str, mode: BindingMode = _TOGGLE) -> Binding:
    return Binding(combo, mode, _DICTATE)


class TestNewKeysParse:
    @pytest.mark.parametrize("digit", list("0123456789"))
    def test_every_digit_parses_to_its_own_key(self, digit):
        assert parse_hotkey_combo(f"ctrl+{digit}") == (
            keys.HotkeyKey(keys.VK_LCONTROL), keys.HotkeyKey(ord(digit)),
        )

    @pytest.mark.parametrize("number", range(13, 25))
    def test_f13_to_f24_parse_and_name_back(self, number):
        (key,) = parse_hotkey_combo(f"F{number}")

        assert key.vk == keys.VK_F1 + number - 1
        assert keys.name_for_vk(key.vk) == f"f{number}"

    @pytest.mark.parametrize(("name", "vk"), [
        ("mouse4", keys.VK_XBUTTON1),
        ("mouse5", keys.VK_XBUTTON2),
        ("mouse_middle", keys.VK_MBUTTON),
    ])
    def test_mouse_buttons_parse_and_name_back(self, name, vk):
        assert parse_hotkey_combo(f"ctrl+{name}")[1].vk == vk
        assert keys.name_for_vk(vk) == name

    @pytest.mark.parametrize("combo", ["esc", "f25", "é", "ctrl+", "", "mouse6", "ctrl+ß"])
    def test_unknown_keys_are_rejected(self, combo):
        with pytest.raises(ValueError, match="Unknown key"):
            parse_hotkey_combo(combo)

    @pytest.mark.parametrize(("vk", "name"), sorted(keys.PUNCTUATION_KEYS.items()))
    def test_every_punctuation_key_parses_in_any_case_and_names_back(self, vk, name):
        assert parse_hotkey_combo(f"ctrl + {name.upper()} ")[1].vk == vk
        assert keys.name_for_vk(vk) == name

    @pytest.mark.parametrize("printed", ["'", ";", "[", "\\", "`", "ä", "😀"])
    def test_a_punctuation_key_is_written_by_name_not_by_the_character_it_prints(self, printed):
        with pytest.raises(ValueError, match="Unknown key"):
            parse_hotkey_combo(f"ctrl+{printed}")

    def test_the_extra_key_on_iso_keyboards_is_still_unsupported(self):
        assert keys.name_for_vk(0xE2) is None


class TestValidation:
    def test_the_default_shortcut_is_valid(self):
        validate_bindings(DEFAULT_BINDINGS)

    def test_escape_alone_is_rejected(self):
        with pytest.raises(BindingError, match="esc"):
            validate_bindings([_dictate("esc")])

    @pytest.mark.parametrize("modifier", ["ctrl", "alt", "shift", "cmd", "ctrl_r"])
    def test_a_single_modifier_alone_is_rejected(self, modifier):
        with pytest.raises(BindingError, match="needs another key"):
            validate_bindings([_dictate(modifier)])

    def test_two_modifiers_together_are_a_valid_hold(self):
        validate_bindings([_dictate("ctrl+cmd", _HOLD)])

    def test_the_copilot_key_followed_by_quote_is_a_valid_shortcut(self):
        validate_bindings([_dictate("cmd+shift+f23+quote")])

    def test_the_same_keys_in_another_order_are_rejected_as_a_duplicate(self):
        with pytest.raises(BindingError, match="used twice"):
            validate_bindings([
                _dictate("ctrl+alt+m"),
                Binding("alt+ctrl+m", BindingMode.PRESS, BindingAction.PASTE_LAST),
            ])

    def test_an_identical_shortcut_for_two_actions_is_rejected(self):
        with pytest.raises(BindingError, match="mouse4 is used twice"):
            validate_bindings([
                _dictate("mouse4", _HOLD),
                Binding("mouse4", BindingMode.PRESS, BindingAction.COPY_LAST),
            ])

    def test_a_shortcut_that_is_the_start_of_another_is_rejected(self):
        with pytest.raises(
            BindingError, match="Pressing ctrl\\+alt\\+m would also trigger ctrl\\+alt",
        ):
            validate_bindings([_dictate("ctrl+alt+m"), _dictate("ctrl+alt", _HOLD)])

    @pytest.mark.parametrize(("inner", "outer"), [
        ("mouse4", "ctrl+mouse4"),
        ("alt+m", "ctrl+alt+m"),
        ("f13", "shift+f13"),
    ])
    def test_a_shortcut_that_fires_inside_another_is_rejected(self, inner, outer):
        with pytest.raises(BindingError, match=re.escape(f"would also trigger {inner}.")):
            validate_bindings([_dictate(outer), _dictate(inner, _HOLD)])

    def test_shortcuts_sharing_modifiers_but_not_their_last_key_are_fine(self):
        validate_bindings([
            _dictate("ctrl+alt+m"),
            _dictate("ctrl+mouse4", _HOLD),
            Binding("ctrl+alt+v", BindingMode.PRESS, BindingAction.PASTE_LAST),
            Binding("ctrl+alt+c", BindingMode.PRESS, BindingAction.COPY_LAST),
            _dictate("ctrl+alt+e", _HOLD),
            _dictate("f13"),
        ])

    def test_a_key_repeated_inside_one_shortcut_is_rejected(self):
        with pytest.raises(BindingError, match="repeats a key"):
            validate_bindings([_dictate("ctrl+m+ctrl")])

    def test_only_optional_shortcuts_are_rejected(self):
        with pytest.raises(BindingError, match="at least one dictation shortcut"):
            validate_bindings([
                Binding("f13", BindingMode.PRESS, BindingAction.PASTE_LAST),
            ])


class TestParsingSavedBindings:
    def test_bindings_round_trip_through_config(self):
        bindings = (
            _dictate("ctrl+shift+alt+m"),
            _dictate("mouse4", _HOLD),
            Binding("f13", BindingMode.PRESS, BindingAction.PASTE_LAST),
            Binding("mouse5", BindingMode.PRESS, BindingAction.COPY_LAST),
        )
        config = FakeConfig({"hotkey.bindings": bindings_to_config(bindings)})

        assert parse_bindings(config) == bindings

    def test_a_config_without_bindings_gets_the_default(self):
        assert parse_bindings(FakeConfig()) == DEFAULT_BINDINGS

    def test_broken_entries_are_skipped_and_the_rest_kept(self):
        rows: list[object] = [
            {"keys": "ctrl+ü", "mode": "toggle", "action": "dictate"},
            {"keys": "ctrl+alt+e", "mode": "hold", "action": "edit_selection"},
            {"keys": "", "mode": "toggle", "action": "paste_last"},
            {"keys": "   ", "mode": "hold", "action": "copy_last"},
            {"keys": "f12", "mode": "toggle", "action": "summon_demons"},
            {"keys": "f12", "mode": "toggle", "action": ["dictate"]},
            {"keys": 12, "mode": "toggle", "action": "dictate"},
            "ctrl+alt+m",
            None,
            {"keys": "mouse4", "mode": "hold", "action": "dictate"},
        ]
        config = FakeConfig({"hotkey.bindings": rows})

        assert parse_bindings(config) == (_dictate("mouse4", _HOLD),)

    def test_modes_are_made_to_fit_their_action(self):
        rows: list[object] = [
            {"keys": "f13", "mode": "hold", "action": "paste_last"},
            {"keys": "f14", "mode": "press", "action": "dictate"},
            {"keys": "f15", "mode": "push_to_talk", "action": "dictate"},
            {"keys": "f16", "action": "dictate"},
        ]

        parsed = parse_bindings(FakeConfig({"hotkey.bindings": rows}))

        assert [binding.mode for binding in parsed] == [
            BindingMode.PRESS, _TOGGLE, _TOGGLE, _TOGGLE,
        ]

    def test_thousands_of_saved_rows_with_duplicates_and_junk_parse_quickly(self):
        rows: list[object] = []
        for index in range(3000):
            rows.append({"keys": f"ctrl+f{index % 24 + 1}", "mode": "hold", "action": "dictate"})
            rows.append({"keys": "", "mode": "toggle", "action": "dictate"})
            rows.append({"keys": f"ctrl+{index}", "mode": "toggle", "action": "dictate"})

        parsed = parse_bindings(FakeConfig({"hotkey.bindings": rows}))

        assert len(parsed) == 3000 + 10
        assert parsed[0] == _dictate("ctrl+f1", _HOLD)

    def test_a_bindings_value_that_is_not_a_list_turns_every_shortcut_off(self):
        assert parse_bindings(FakeConfig({"hotkey.bindings": "ctrl+alt+m"})) == ()
