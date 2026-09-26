"""Hotkey activation modes - toggle vs hold - and their
interaction with in-flight transcription."""

from tests.acceptance.app_fixture import VoiceToTextApp
from tests.fakes.hotkeys import FakeClock, FakeTimers
from voicetotext.core.state_machine import AppState
from voicetotext.hotkeys import keys
from voicetotext.hotkeys.toggle_strategy import ToggleStrategy
from voicetotext.hotkeys.hold_strategy import HoldStrategy


_CTRL = keys.HotkeyKey(keys.VK_LCONTROL)
_ALT = keys.HotkeyKey(keys.VK_LMENU)
_M = keys.from_char("m")
_W = keys.from_char("w")
_MOUSE4 = keys.HotkeyKey(keys.VK_XBUTTON1)


class _LockableHold:
    def __init__(self, app: VoiceToTextApp) -> None:
        self.clock = FakeClock()
        self.timers = FakeTimers()
        self.strategy = HoldStrategy(
            (_MOUSE4,), double_tap_locks=True,
            clock=self.clock, make_timer=self.timers.make,
        )
        self.strategy.set_callbacks(app.press_hotkey, app.release_hotkey)

    def tap(self) -> None:
        self.strategy.on_key_press(_MOUSE4)
        self.clock.now += 0.09
        self.strategy.on_key_release(_MOUSE4)
        self.clock.now += 0.09

    def let_the_window_pass(self) -> None:
        self.clock.now += 1.0
        self.timers.fire_all()


class TestDoubleTapLock:
    def test_a_double_tap_keeps_recording_until_the_next_press(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("hands free")
        hold = _LockableHold(voice_to_text_app)

        hold.tap()
        hold.tap()
        hold.let_the_window_pass()

        assert voice_to_text_app.state == AppState.RECORDING

        hold.tap()
        hold.let_the_window_pass()

        assert voice_to_text_app.typed_text == "hands free"
        assert voice_to_text_app.state == AppState.IDLE

    def test_after_a_locked_recording_is_cancelled_the_next_press_starts_a_new_one(
        self, voice_to_text_app: VoiceToTextApp
    ):
        hold = _LockableHold(voice_to_text_app)
        hold.tap()
        hold.tap()
        voice_to_text_app.press_escape()
        assert voice_to_text_app.state == AppState.IDLE

        hold.strategy.on_key_press(_MOUSE4)

        assert voice_to_text_app.state == AppState.RECORDING

    def test_a_single_tap_records_only_until_the_window_closes(
        self, voice_to_text_app: VoiceToTextApp
    ):
        hold = _LockableHold(voice_to_text_app)

        hold.tap()
        assert voice_to_text_app.state == AppState.RECORDING
        hold.let_the_window_pass()

        assert voice_to_text_app.state == AppState.IDLE


class TestPushToTalk:
    def test_push_to_talk_starts_on_press_and_transcribes_on_release(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("push to talk")

        voice_to_text_app.press_hotkey()
        assert voice_to_text_app.state == AppState.RECORDING

        voice_to_text_app.release_hotkey()
        assert voice_to_text_app.typed_text == "push to talk"
        assert voice_to_text_app.state == AppState.IDLE

    def test_release_without_prior_press_does_nothing(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.release_hotkey()

        assert voice_to_text_app.state == AppState.IDLE
        assert voice_to_text_app.typed_text is None


class TestToggle:
    def test_toggle_requires_two_presses_to_complete_a_recording(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("toggle works")

        voice_to_text_app.press_hotkey()
        assert voice_to_text_app.state == AppState.RECORDING
        assert voice_to_text_app.typed_text is None

        voice_to_text_app.press_hotkey()
        assert voice_to_text_app.typed_text == "toggle works"
        assert voice_to_text_app.state == AppState.IDLE

    def test_toggling_twice_produces_two_independent_transcriptions(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("first")
        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        voice_to_text_app.user_speaks("second")
        voice_to_text_app.press_hotkey()
        voice_to_text_app.press_hotkey()

        assert voice_to_text_app.all_typed_text == ["first", "second"]


class TestExactSequenceMatching:
    def test_toggle_activates_when_keys_pressed_in_exact_order(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("exact order")
        strategy = ToggleStrategy(keys=(_CTRL, _ALT, _M))
        strategy.set_callbacks(
            voice_to_text_app.press_hotkey, voice_to_text_app.press_hotkey
        )

        strategy.on_key_press(_CTRL)
        strategy.on_key_press(_ALT)
        strategy.on_key_press(_M)

        assert voice_to_text_app.state == AppState.RECORDING

    def test_toggle_does_not_activate_when_extra_key_interleaved(
        self, voice_to_text_app: VoiceToTextApp
    ):
        strategy = ToggleStrategy(keys=(_CTRL, _ALT, _M))
        strategy.set_callbacks(
            voice_to_text_app.press_hotkey, voice_to_text_app.press_hotkey
        )

        strategy.on_key_press(_CTRL)
        strategy.on_key_press(_W)
        strategy.on_key_press(_ALT)
        strategy.on_key_press(_M)

        assert voice_to_text_app.state == AppState.IDLE

    def test_push_to_talk_activates_and_deactivates_with_exact_sequence(
        self, voice_to_text_app: VoiceToTextApp
    ):
        voice_to_text_app.user_speaks("held sequence")
        strategy = HoldStrategy(keys=(_CTRL, _M))
        strategy.set_callbacks(
            voice_to_text_app.press_hotkey, voice_to_text_app.release_hotkey
        )

        strategy.on_key_press(_CTRL)
        strategy.on_key_press(_M)
        assert voice_to_text_app.state == AppState.RECORDING

        strategy.on_key_release(_M)
        assert voice_to_text_app.typed_text == "held sequence"
        assert voice_to_text_app.state == AppState.IDLE
