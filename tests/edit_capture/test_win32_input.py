from unittest.mock import patch

from voicetotext.edit_capture import win32_input as wi
from voicetotext.edit_capture.win32_input import (
    clicked_since_last_check,
    key_character,
)


_VK_A = 0x41
_HIGH_BIT_ONLY = 0x8000  # button held down, but not newly pressed


def _translates_to(text: str):
    def _call(vk, scan_code, state, buffer, length, flags, layout):
        buffer.value = text
        return len(text)

    return _call


def _returns(code: int):
    def _call(vk, scan_code, state, buffer, length, flags, layout):
        return code

    return _call


@patch.object(wi, "_user32")
class TestKeyCharacterTranslatesAKey:
    def test_should_return_the_translated_character(self, mock_user32):
        mock_user32.ToUnicodeEx.side_effect = _translates_to("a")

        assert key_character(_VK_A, shift=False, ctrl=False, alt=False) == "a"

    def test_should_return_empty_for_a_dead_key(self, mock_user32):
        mock_user32.ToUnicodeEx.side_effect = _returns(-1)

        assert key_character(0xDD, shift=False, ctrl=False, alt=False) == ""

    def test_should_return_empty_when_the_layout_translates_nothing(
        self, mock_user32,
    ):
        mock_user32.ToUnicodeEx.side_effect = _returns(0)

        assert key_character(0x70, shift=False, ctrl=False, alt=False) == ""

    def test_should_return_empty_for_a_control_character(self, mock_user32):
        # Ctrl+A translates to \x01, which types nothing into the box.
        mock_user32.ToUnicodeEx.side_effect = _translates_to("\x01")

        assert key_character(_VK_A, shift=False, ctrl=True, alt=False) == ""


@patch.object(wi, "_user32")
class TestKeyCharacterAsksWindowsSafely:
    def test_should_build_key_state_from_the_given_modifiers(self, mock_user32):
        mock_user32.ToUnicodeEx.side_effect = _translates_to("A")

        key_character(_VK_A, shift=True, ctrl=False, alt=True)

        state = mock_user32.ToUnicodeEx.call_args.args[2]
        assert state[wi._VK_SHIFT] == wi._KEY_DOWN_FLAG
        assert state[wi._VK_MENU] == wi._KEY_DOWN_FLAG
        assert state[wi._VK_CONTROL] == 0
        assert state[_VK_A] == 0

    def test_should_not_read_the_live_keyboard_state(self, mock_user32):
        mock_user32.ToUnicodeEx.side_effect = _translates_to("a")

        key_character(_VK_A, shift=False, ctrl=False, alt=False)

        mock_user32.GetKeyboardState.assert_not_called()

    def test_should_ask_without_disturbing_the_kernel_dead_key_state(
        self, mock_user32,
    ):
        mock_user32.ToUnicodeEx.side_effect = _translates_to("a")

        key_character(_VK_A, shift=False, ctrl=False, alt=False)

        flags = mock_user32.ToUnicodeEx.call_args.args[5]
        assert flags & wi._LEAVE_KERNEL_KEY_STATE_ALONE

    def test_should_use_the_layout_of_the_foreground_thread(self, mock_user32):
        mock_user32.GetWindowThreadProcessId.return_value = 77
        mock_user32.GetKeyboardLayout.return_value = 0x0409
        mock_user32.ToUnicodeEx.side_effect = _translates_to("a")

        key_character(_VK_A, shift=False, ctrl=False, alt=False)

        mock_user32.GetKeyboardLayout.assert_called_once_with(77)
        assert mock_user32.ToUnicodeEx.call_args.args[6] == 0x0409


@patch.object(wi, "_user32")
class TestClickedSinceLastCheck:
    def test_should_report_a_click_when_the_pressed_bit_is_set(self, mock_user32):
        mock_user32.GetAsyncKeyState.return_value = wi._PRESSED_SINCE_LAST_CALL

        assert clicked_since_last_check() is True

    def test_should_report_no_click_when_no_button_reports_one(self, mock_user32):
        mock_user32.GetAsyncKeyState.return_value = 0

        assert clicked_since_last_check() is False

    def test_should_ignore_a_button_that_is_merely_held_down(self, mock_user32):
        mock_user32.GetAsyncKeyState.return_value = _HIGH_BIT_ONLY

        assert clicked_since_last_check() is False

    def test_should_report_a_click_from_any_of_the_three_buttons(self, mock_user32):
        for button in wi._MOUSE_BUTTONS:
            mock_user32.GetAsyncKeyState.side_effect = (
                lambda vk, pressed=button: (
                    wi._PRESSED_SINCE_LAST_CALL if vk == pressed else 0
                )
            )

            assert clicked_since_last_check() is True

    def test_should_read_every_button_even_after_one_reports_a_click(
        self, mock_user32,
    ):
        # Reading the bit clears it, so a button left unread would report
        # its stale click on the next call.
        mock_user32.GetAsyncKeyState.return_value = wi._PRESSED_SINCE_LAST_CALL

        clicked_since_last_check()

        read = [call.args[0] for call in mock_user32.GetAsyncKeyState.call_args_list]
        assert read == list(wi._MOUSE_BUTTONS)
