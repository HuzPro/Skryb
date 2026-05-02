from unittest.mock import MagicMock

from pynput.keyboard import Key, KeyCode

from voicetotext.hotkeys.toggle_strategy import ToggleStrategy


def _make_strategy(*keys):
    strategy = ToggleStrategy(keys=keys)
    on_activate = MagicMock()
    on_deactivate = MagicMock()
    strategy.set_callbacks(on_activate, on_deactivate)
    return strategy, on_activate, on_deactivate


class TestToggleStrategyActivation:
    def test_should_fire_activate_on_first_combo_press(self):
        strategy, on_activate, _ = _make_strategy(Key.ctrl_l, KeyCode.from_char("m"))

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("m"))

        on_activate.assert_called_once()

    def test_should_fire_deactivate_on_second_combo_press(self):
        strategy, _, on_deactivate = _make_strategy(Key.ctrl_l, KeyCode.from_char("m"))

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("m"))
        strategy.on_key_release(KeyCode.from_char("m"))
        strategy.on_key_release(Key.ctrl_l)

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("m"))

        on_deactivate.assert_called_once()

    def test_should_alternate_between_activate_and_deactivate(self):
        strategy, on_activate, on_deactivate = _make_strategy(
            Key.ctrl_l, KeyCode.from_char("m")
        )

        for _ in range(4):
            strategy.on_key_press(Key.ctrl_l)
            strategy.on_key_press(KeyCode.from_char("m"))
            strategy.on_key_release(KeyCode.from_char("m"))
            strategy.on_key_release(Key.ctrl_l)

        assert on_activate.call_count == 2
        assert on_deactivate.call_count == 2


class TestToggleStrategyExactSequence:
    def test_should_not_activate_when_keys_pressed_in_wrong_order(self):
        strategy, on_activate, _ = _make_strategy(Key.ctrl_l, KeyCode.from_char("m"))

        strategy.on_key_press(KeyCode.from_char("m"))
        strategy.on_key_press(Key.ctrl_l)

        on_activate.assert_not_called()

    def test_should_not_activate_when_extra_key_interleaved(self):
        strategy, on_activate, _ = _make_strategy(Key.ctrl_l, KeyCode.from_char("m"))

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("x"))
        strategy.on_key_press(KeyCode.from_char("m"))

        on_activate.assert_not_called()

    def test_should_activate_five_key_sequence_in_exact_order(self):
        k1 = KeyCode.from_char("1")
        k2 = KeyCode.from_char("2")
        k3 = KeyCode.from_char("3")
        k4 = KeyCode.from_char("4")
        k5 = KeyCode.from_char("5")
        strategy, on_activate, _ = _make_strategy(Key.alt_l, k1, k2, k3, k4, k5)

        strategy.on_key_press(Key.alt_l)
        strategy.on_key_press(k1)
        strategy.on_key_press(k2)
        strategy.on_key_press(k3)
        strategy.on_key_press(k4)
        strategy.on_key_press(k5)

        on_activate.assert_called_once()

    def test_should_allow_retry_after_interleaved_key_resets_sequence(self):
        strategy, on_activate, _ = _make_strategy(Key.ctrl_l, KeyCode.from_char("m"))

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("x"))
        strategy.on_key_release(KeyCode.from_char("x"))
        strategy.on_key_release(Key.ctrl_l)

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("m"))

        on_activate.assert_called_once()


class TestToggleStrategyIgnoresNonMatchingKeys:
    def test_should_not_fire_for_partial_combo(self):
        strategy, on_activate, on_deactivate = _make_strategy(
            Key.ctrl_l, KeyCode.from_char("m")
        )

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_release(Key.ctrl_l)

        on_activate.assert_not_called()
        on_deactivate.assert_not_called()

    def test_should_not_fire_for_unrelated_key(self):
        strategy, on_activate, _ = _make_strategy(
            Key.ctrl_l, KeyCode.from_char("m")
        )

        strategy.on_key_press(KeyCode.from_char("x"))
        strategy.on_key_release(KeyCode.from_char("x"))

        on_activate.assert_not_called()


class TestToggleStrategyKeyNormalization:
    def test_should_activate_when_key_reported_with_vk_instead_of_char(self):
        strategy, on_activate, _ = _make_strategy(
            Key.ctrl_l, Key.alt_l, KeyCode.from_char("m")
        )

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(Key.alt_l)
        strategy.on_key_press(KeyCode.from_vk(0x4D))

        on_activate.assert_called_once()

    def test_should_deactivate_when_vk_key_released(self):
        strategy, _, on_deactivate = _make_strategy(
            Key.ctrl_l, Key.alt_l, KeyCode.from_char("m")
        )

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(Key.alt_l)
        strategy.on_key_press(KeyCode.from_vk(0x4D))
        strategy.on_key_release(KeyCode.from_vk(0x4D))
        strategy.on_key_release(Key.alt_l)
        strategy.on_key_release(Key.ctrl_l)

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(Key.alt_l)
        strategy.on_key_press(KeyCode.from_vk(0x4D))

        on_deactivate.assert_called_once()

    def test_should_activate_when_ctrl_modifies_char_to_control_character(self):
        strategy, on_activate, _ = _make_strategy(
            Key.ctrl_l, Key.alt_l, KeyCode.from_char("m")
        )

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(Key.alt_l)
        strategy.on_key_press(KeyCode(char="\r"))

        on_activate.assert_called_once()

    def test_should_activate_when_key_has_both_char_and_vk(self):
        strategy, on_activate, _ = _make_strategy(
            Key.ctrl_l, Key.alt_l, KeyCode.from_char("m")
        )

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(Key.alt_l)
        strategy.on_key_press(KeyCode(char="\r", vk=0x4D))

        on_activate.assert_called_once()


class TestToggleStrategySingleKey:
    def test_should_work_with_single_key_combo(self):
        strategy, on_activate, on_deactivate = _make_strategy(Key.f12)

        strategy.on_key_press(Key.f12)
        on_activate.assert_called_once()
        strategy.on_key_release(Key.f12)

        strategy.on_key_press(Key.f12)
        on_deactivate.assert_called_once()
