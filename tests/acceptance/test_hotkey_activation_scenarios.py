"""Hotkey activation modes — toggle vs push-to-talk — and their
interaction with in-flight transcription."""

from tests.acceptance.app_fixture import VoiceToTextApp
from voicetotext.core.state_machine import AppState
from voicetotext.hotkeys import keys
from voicetotext.hotkeys.toggle_strategy import ToggleStrategy
from voicetotext.hotkeys.push_to_talk_strategy import PushToTalkStrategy


_CTRL = keys.HotkeyKey(keys.VK_LCONTROL)
_ALT = keys.HotkeyKey(keys.VK_LMENU)
_M = keys.from_char("m")
_W = keys.from_char("w")


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
        strategy = PushToTalkStrategy(keys=(_CTRL, _M))
        strategy.set_callbacks(
            voice_to_text_app.press_hotkey, voice_to_text_app.release_hotkey
        )

        strategy.on_key_press(_CTRL)
        strategy.on_key_press(_M)
        assert voice_to_text_app.state == AppState.RECORDING

        strategy.on_key_release(_M)
        assert voice_to_text_app.typed_text == "held sequence"
        assert voice_to_text_app.state == AppState.IDLE
