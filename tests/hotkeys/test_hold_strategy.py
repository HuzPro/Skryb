import itertools
from unittest.mock import MagicMock

import pytest

from tests.fakes.hotkeys import FakeClock, FakeTimers
from voicetotext.hotkeys import keys
from voicetotext.hotkeys.hold_strategy import HoldStrategy


CTRL = keys.HotkeyKey(keys.VK_LCONTROL)
ALT = keys.HotkeyKey(keys.VK_LMENU)
M = keys.from_char("m")
X = keys.from_char("x")
W = keys.from_char("w")
F12 = keys.HotkeyKey(0x7B)


def _make_strategy(*combo):
    strategy = HoldStrategy(keys=combo)
    on_activate = MagicMock()
    on_deactivate = MagicMock()
    strategy.set_callbacks(on_activate, on_deactivate)
    return strategy, on_activate, on_deactivate


class TestHoldActivation:
    def test_should_fire_activate_when_all_keys_pressed_in_order(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(M)

        on_activate.assert_called_once()

    def test_should_fire_deactivate_when_any_key_released(self):
        strategy, _, on_deactivate = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(M)
        strategy.on_key_release(M)

        on_deactivate.assert_called_once()

    def test_should_support_rapid_press_release_cycles(self):
        strategy, on_activate, on_deactivate = _make_strategy(F12)

        for _ in range(3):
            strategy.on_key_press(F12)
            strategy.on_key_release(F12)

        assert on_activate.call_count == 3
        assert on_deactivate.call_count == 3


class TestHoldMatchesHookVkEvents:
    def test_should_match_when_target_is_from_char_and_event_is_from_vk(self):
        strategy, on_activate, _ = _make_strategy(CTRL, ALT, M)

        strategy.on_key_press(keys.from_vk(keys.VK_LCONTROL))
        strategy.on_key_press(keys.from_vk(keys.VK_LMENU))
        strategy.on_key_press(keys.from_vk(0x4D))

        on_activate.assert_called_once()

    def test_should_deactivate_when_vk_key_released(self):
        strategy, _, on_deactivate = _make_strategy(CTRL, ALT, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(ALT)
        strategy.on_key_press(keys.from_vk(0x4D))
        strategy.on_key_release(keys.from_vk(0x4D))

        on_deactivate.assert_called_once()


class TestHoldExactSequence:
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

    def test_should_not_activate_five_key_sequence_with_interleave(self):
        digits = tuple(keys.from_char(d) for d in "12345")
        strategy, on_activate, _ = _make_strategy(ALT, *digits)

        strategy.on_key_press(ALT)
        strategy.on_key_press(digits[0])
        strategy.on_key_press(digits[1])
        strategy.on_key_press(W)
        for digit in digits[2:]:
            strategy.on_key_press(digit)

        on_activate.assert_not_called()

    def test_should_allow_retry_after_interleaved_key_resets_sequence(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(X)
        strategy.on_key_release(X)
        strategy.on_key_release(CTRL)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(M)

        on_activate.assert_called_once()


class TestHoldIgnoresNonMatchingKeys:
    def test_should_not_fire_for_partial_combo(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_release(CTRL)

        on_activate.assert_not_called()

    def test_should_not_fire_deactivate_if_never_activated(self):
        strategy, _, on_deactivate = _make_strategy(F12)

        strategy.on_key_release(F12)

        on_deactivate.assert_not_called()


class TestHoldIsNotFooledByOtherKeys:
    def test_two_thousand_autorepeat_key_downs_never_retrigger_a_hold(self):
        strategy, on_activate, on_deactivate = _make_strategy(CTRL, ALT, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(ALT)
        for _ in range(2000):
            strategy.on_key_press(M)
            strategy.on_key_press(CTRL)
        strategy.on_key_release(M)

        on_activate.assert_called_once()
        on_deactivate.assert_called_once()

    @pytest.mark.parametrize("order", list(itertools.permutations((CTRL, ALT, M))))
    def test_every_release_order_of_a_three_key_hold_releases_exactly_once(self, order):
        strategy, on_activate, on_deactivate = _make_strategy(CTRL, ALT, M)

        for key in (CTRL, ALT, M):
            strategy.on_key_press(key)
        for key in order:
            strategy.on_key_release(key)

        on_activate.assert_called_once()
        on_deactivate.assert_called_once()

    def test_typing_while_holding_does_not_let_go(self):
        strategy, _, on_deactivate = _make_strategy(CTRL, W)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(W)
        for letter in "hello there 42":
            key = keys.HotkeyKey(keys.VK_SPACE) if letter == " " else keys.from_char(letter)
            strategy.on_key_press(key)
            strategy.on_key_release(key)

        on_deactivate.assert_not_called()


def _lockable(*combo):
    clock, timers = FakeClock(), FakeTimers()
    strategy = HoldStrategy(
        combo, double_tap_locks=True, clock=clock, make_timer=timers.make,
    )
    on_activate, on_deactivate = MagicMock(), MagicMock()
    strategy.set_callbacks(on_activate, on_deactivate)
    return strategy, clock, timers, on_activate, on_deactivate


def _press_and_release(strategy, clock, held_for, *combo):
    for key in combo:
        strategy.on_key_press(key)
    clock.now += held_for
    for key in reversed(combo):
        strategy.on_key_release(key)


class TestDoubleTapLocksAHoldOn:
    def test_a_double_tap_within_the_window_locks_and_the_next_press_toggles_off(self):
        strategy, clock, timers, on_activate, on_deactivate = _lockable(CTRL, M)

        _press_and_release(strategy, clock, 0.12, CTRL, M)
        clock.now += 0.15
        _press_and_release(strategy, clock, 0.10, CTRL, M)
        timers.fire_all()
        clock.now += 30.0

        on_activate.assert_called_once()
        on_deactivate.assert_not_called()

        _press_and_release(strategy, clock, 0.10, CTRL, M)
        timers.fire_all()

        assert on_activate.call_count == 2
        on_deactivate.assert_not_called()

        _press_and_release(strategy, clock, 1.0, CTRL, M)

        assert on_activate.call_count == 3
        on_deactivate.assert_called_once()

    def test_a_second_press_after_the_window_does_not_lock(self):
        strategy, clock, timers, on_activate, on_deactivate = _lockable(F12)

        _press_and_release(strategy, clock, 0.1, F12)
        clock.now += 0.5
        timers.fire_all()
        strategy.on_key_press(F12)
        clock.now += 2.0
        strategy.on_key_release(F12)

        assert on_activate.call_count == 2
        assert on_deactivate.call_count == 2

    def test_a_press_held_past_the_window_is_an_ordinary_hold(self):
        strategy, clock, timers, _, on_deactivate = _lockable(CTRL, M)

        _press_and_release(strategy, clock, 0.41, CTRL, M)

        on_deactivate.assert_called_once()
        assert timers.pending == []

    def test_a_single_quick_tap_stops_when_the_window_closes(self):
        strategy, clock, timers, _, on_deactivate = _lockable(F12)

        _press_and_release(strategy, clock, 0.1, F12)
        on_deactivate.assert_not_called()
        assert timers.pending[0][0] == pytest.approx(0.3)

        timers.fire_all()

        on_deactivate.assert_called_once()

    def test_holding_the_modifier_and_double_tapping_the_last_key_also_locks(self):
        strategy, clock, timers, on_activate, on_deactivate = _lockable(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_press(M)
        clock.now += 0.08
        strategy.on_key_release(M)
        clock.now += 0.08
        strategy.on_key_press(M)
        strategy.on_key_release(M)
        strategy.on_key_release(CTRL)
        timers.fire_all()

        on_activate.assert_called_once()
        on_deactivate.assert_not_called()

    def test_without_the_setting_a_quick_tap_stops_at_once(self):
        strategy, on_activate, on_deactivate = _make_strategy(F12)

        strategy.on_key_press(F12)
        strategy.on_key_release(F12)

        on_activate.assert_called_once()
        on_deactivate.assert_called_once()
