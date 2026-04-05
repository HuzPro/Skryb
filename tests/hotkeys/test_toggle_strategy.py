from unittest.mock import MagicMock

from pynput.keyboard import Key, KeyCode

from voicetotext.hotkeys.toggle_strategy import ToggleStrategy


def _make_strategy(*keys):
    strategy = ToggleStrategy(keys=set(keys))
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

        # First combo press (activate)
        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("m"))
        strategy.on_key_release(KeyCode.from_char("m"))
        strategy.on_key_release(Key.ctrl_l)

        # Second combo press (deactivate)
        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("m"))

        on_deactivate.assert_called_once()

    def test_should_alternate_between_activate_and_deactivate(self):
        strategy, on_activate, on_deactivate = _make_strategy(
            Key.ctrl_l, KeyCode.from_char("m")
        )

        for i in range(4):
            strategy.on_key_press(Key.ctrl_l)
            strategy.on_key_press(KeyCode.from_char("m"))
            strategy.on_key_release(KeyCode.from_char("m"))
            strategy.on_key_release(Key.ctrl_l)

        assert on_activate.call_count == 2
        assert on_deactivate.call_count == 2


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


class TestToggleStrategySingleKey:
    def test_should_work_with_single_key_combo(self):
        strategy, on_activate, on_deactivate = _make_strategy(Key.f12)

        strategy.on_key_press(Key.f12)
        on_activate.assert_called_once()
        strategy.on_key_release(Key.f12)

        strategy.on_key_press(Key.f12)
        on_deactivate.assert_called_once()
