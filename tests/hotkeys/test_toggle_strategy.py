from unittest.mock import MagicMock

from skryb.hotkeys import keys
from skryb.hotkeys.toggle_strategy import ToggleStrategy


CTRL = keys.HotkeyKey(keys.VK_LCONTROL)
ALT = keys.HotkeyKey(keys.VK_LMENU)
M = keys.from_char("m")
X = keys.from_char("x")
F12 = keys.HotkeyKey(0x7B)


def _make_strategy(*combo):
    strategy = ToggleStrategy(keys=combo)
    on_activate = MagicMock()
    on_deactivate = MagicMock()
    strategy.set_callbacks(on_activate, on_deactivate)
    return strategy, on_activate, on_deactivate


class TestToggleStrategyActivation:
    def test_should_fire_activate_on_first_combo_press(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(M)

        on_activate.assert_called_once()

    def test_every_combo_press_is_one_toggle_request(self):
        strategy, on_activate, on_deactivate = _make_strategy(CTRL, M)

        for _ in range(4):
            strategy.on_key_press(CTRL)
            strategy.on_key_press(M)
            strategy.on_key_release(M)
            strategy.on_key_release(CTRL)

        assert on_activate.call_count == 4
        on_deactivate.assert_not_called()

    def test_holding_the_combo_down_asks_only_once(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        for _ in range(30):
            strategy.on_key_press(M)

        on_activate.assert_called_once()


class TestToggleStrategyExactSequence:
    def test_should_not_activate_when_keys_pressed_in_wrong_order(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(M)
        strategy.on_key_press(CTRL)

        on_activate.assert_not_called()

    def test_should_not_activate_when_extra_key_interleaved(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(X)
        strategy.on_key_press(M)

        on_activate.assert_not_called()

    def test_should_activate_five_key_sequence_in_exact_order(self):
        digits = tuple(keys.from_char(d) for d in "12345")
        strategy, on_activate, _ = _make_strategy(ALT, *digits)

        strategy.on_key_press(ALT)
        for digit in digits:
            strategy.on_key_press(digit)

        on_activate.assert_called_once()

    def test_should_allow_retry_after_interleaved_key_resets_sequence(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(X)
        strategy.on_key_release(X)
        strategy.on_key_release(CTRL)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(M)

        on_activate.assert_called_once()


class TestToggleStrategyIgnoresNonMatchingKeys:
    def test_should_not_fire_for_partial_combo(self):
        strategy, on_activate, on_deactivate = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_release(CTRL)

        on_activate.assert_not_called()
        on_deactivate.assert_not_called()

    def test_should_not_fire_for_unrelated_key(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(X)
        strategy.on_key_release(X)

        on_activate.assert_not_called()


class TestToggleStrategyMatchesHookVkEvents:
    def test_should_match_when_target_is_from_char_and_event_is_from_vk(self):
        # A hotkey configured as "ctrl+alt+m" parses its letter via
        # ``from_char``; the hook delivers that letter as vk 0x4D. Equality
        # by vk must bridge the two.
        strategy, on_activate, _ = _make_strategy(CTRL, ALT, M)

        strategy.on_key_press(keys.from_vk(keys.VK_LCONTROL))
        strategy.on_key_press(keys.from_vk(keys.VK_LMENU))
        strategy.on_key_press(keys.from_vk(0x4D))

        on_activate.assert_called_once()


class TestToggleStrategySingleKey:
    def test_should_work_with_single_key_combo(self):
        strategy, on_activate, _ = _make_strategy(F12)

        strategy.on_key_press(F12)
        strategy.on_key_release(F12)
        strategy.on_key_press(F12)

        assert on_activate.call_count == 2
