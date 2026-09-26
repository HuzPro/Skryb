import ctypes
from unittest.mock import patch

import pytest

from voicetotext.core.exceptions import OutputError
from voicetotext.output import win32_key_sender as wks
from voicetotext.output.key_sender import KeySender
from voicetotext.output.paste_method import PasteMethod
from voicetotext.output.win32_key_sender import Win32KeySender


_VK_CONTROL = 0x11
_VK_V = 0x56
_KEYEVENTF_KEYUP = 0x0002


class TestWin32KeySenderSendCtrlV:
    @patch.object(wks, "_user32")
    def test_should_inject_ctrl_down_v_down_v_up_ctrl_up(self, mock_user32):
        mock_user32.SendInput.return_value = 4

        Win32KeySender().send_paste(PasteMethod.CTRL_V, "")

        mock_user32.SendInput.assert_called_once()
        count, events, _ = mock_user32.SendInput.call_args.args
        assert count == 4
        sequence = [
            (events[i].union.ki.wVk, events[i].union.ki.dwFlags) for i in range(4)
        ]
        assert sequence == [
            (_VK_CONTROL, 0),
            (_VK_V, 0),
            (_VK_V, _KEYEVENTF_KEYUP),
            (_VK_CONTROL, _KEYEVENTF_KEYUP),
        ]

    @patch.object(wks, "_user32")
    def test_should_tag_every_injected_event_as_ours(self, mock_user32):
        mock_user32.SendInput.return_value = 4

        Win32KeySender().send_paste(PasteMethod.CTRL_V, "")

        _, events, _ = mock_user32.SendInput.call_args.args
        tags = [events[i].union.ki.dwExtraInfo for i in range(4)]
        assert tags == [wks.VOICETOTEXT_INPUT_TAG] * 4

    def test_should_keep_the_tag_inside_thirty_two_bits(self):
        # dwExtraInfo is an ULONG_PTR; a wider value would not survive a
        # 32-bit build.
        assert 0 < wks.VOICETOTEXT_INPUT_TAG <= 0xFFFFFFFF

    @patch.object(wks, "_user32")
    def test_should_pass_the_real_input_struct_size(self, mock_user32):
        mock_user32.SendInput.return_value = 4

        Win32KeySender().send_paste(PasteMethod.CTRL_V, "")

        _, _, size = mock_user32.SendInput.call_args.args
        assert size == ctypes.sizeof(wks._Input)

    @patch.object(wks, "_user32")
    def test_should_raise_when_send_input_injects_nothing(self, mock_user32):
        mock_user32.SendInput.return_value = 0

        with pytest.raises(OutputError):
            Win32KeySender().send_paste(PasteMethod.CTRL_V, "")


class TestWin32KeySenderIsAKeySender:
    def test_should_satisfy_key_sender_protocol(self):
        assert isinstance(Win32KeySender(), KeySender)


class TestWin32KeySenderPasteMethods:
    @patch.object(wks, "_user32")
    def test_shift_insert_marks_insert_as_the_navigation_key(self, mock_user32):
        mock_user32.SendInput.return_value = 4

        Win32KeySender().send_paste(PasteMethod.SHIFT_INSERT, "")

        _, events, _ = mock_user32.SendInput.call_args.args
        sequence = [
            (events[i].union.ki.wVk, events[i].union.ki.dwFlags) for i in range(4)
        ]
        assert sequence == [
            (0x10, 0),
            (0x2D, 0x0001),
            (0x2D, 0x0001 | _KEYEVENTF_KEYUP),
            (0x10, _KEYEVENTF_KEYUP),
        ]

    @patch.object(wks, "_user32")
    def test_typing_sends_every_character_as_tagged_unicode_keystrokes(
        self, mock_user32,
    ):
        text = ("Grüße 🎙️ naïve café, 東京 → done.\n" * 120)[:5000]
        typed: list[int] = []

        def send_input(count, events, size):
            for index in range(count):
                ki = events[index].union.ki
                assert ki.dwExtraInfo == wks.VOICETOTEXT_INPUT_TAG
                assert ki.dwFlags & 0x0004
                if not ki.dwFlags & _KEYEVENTF_KEYUP:
                    typed.append(ki.wScan)
            return count

        mock_user32.SendInput.side_effect = send_input

        Win32KeySender().send_paste(PasteMethod.TYPE, text)

        units = b"".join(unit.to_bytes(2, "little") for unit in typed)
        assert units.decode("utf-16-le") == text
