from unittest.mock import MagicMock

from pynput.keyboard import Key, KeyCode

from voicetotext.hotkeys.push_to_talk_strategy import PushToTalkStrategy


def _make_strategy(*keys):
    strategy = PushToTalkStrategy(keys=set(keys))
    on_activate = MagicMock()
    on_deactivate = MagicMock()
    strategy.set_callbacks(on_activate, on_deactivate)
    return strategy, on_activate, on_deactivate


class TestPushToTalkActivation:
    def test_should_fire_activate_when_all_keys_pressed(self):
        strategy, on_activate, _ = _make_strategy(Key.ctrl_l, KeyCode.from_char("m"))

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("m"))

        on_activate.assert_called_once()

    def test_should_fire_deactivate_when_any_key_released(self):
        strategy, _, on_deactivate = _make_strategy(
            Key.ctrl_l, KeyCode.from_char("m")
        )

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_press(KeyCode.from_char("m"))
        strategy.on_key_release(KeyCode.from_char("m"))

        on_deactivate.assert_called_once()

    def test_should_support_rapid_press_release_cycles(self):
        strategy, on_activate, on_deactivate = _make_strategy(Key.f12)

        for _ in range(3):
            strategy.on_key_press(Key.f12)
            strategy.on_key_release(Key.f12)

        assert on_activate.call_count == 3
        assert on_deactivate.call_count == 3


class TestPushToTalkIgnoresNonMatchingKeys:
    def test_should_not_fire_for_partial_combo(self):
        strategy, on_activate, _ = _make_strategy(
            Key.ctrl_l, KeyCode.from_char("m")
        )

        strategy.on_key_press(Key.ctrl_l)
        strategy.on_key_release(Key.ctrl_l)

        on_activate.assert_not_called()

    def test_should_not_fire_deactivate_if_never_activated(self):
        strategy, _, on_deactivate = _make_strategy(Key.f12)

        strategy.on_key_release(Key.f12)

        on_deactivate.assert_not_called()
