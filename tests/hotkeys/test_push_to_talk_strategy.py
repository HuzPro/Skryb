from unittest.mock import MagicMock

from voicetotext.hotkeys import keys
from voicetotext.hotkeys.push_to_talk_strategy import PushToTalkStrategy


CTRL = keys.HotkeyKey(keys.VK_LCONTROL)
ALT = keys.HotkeyKey(keys.VK_LMENU)
M = keys.from_char("m")
X = keys.from_char("x")
W = keys.from_char("w")
F12 = keys.HotkeyKey(0x7B)


def _make_strategy(*combo):
    strategy = PushToTalkStrategy(keys=combo)
    on_activate = MagicMock()
    on_deactivate = MagicMock()
    strategy.set_callbacks(on_activate, on_deactivate)
    return strategy, on_activate, on_deactivate


class TestPushToTalkActivation:
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


class TestPushToTalkMatchesHookVkEvents:
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


class TestPushToTalkExactSequence:
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


class TestPushToTalkIgnoresNonMatchingKeys:
    def test_should_not_fire_for_partial_combo(self):
        strategy, on_activate, _ = _make_strategy(CTRL, M)

        strategy.on_key_press(CTRL)
        strategy.on_key_release(CTRL)

        on_activate.assert_not_called()

    def test_should_not_fire_deactivate_if_never_activated(self):
        strategy, _, on_deactivate = _make_strategy(F12)

        strategy.on_key_release(F12)

        on_deactivate.assert_not_called()
