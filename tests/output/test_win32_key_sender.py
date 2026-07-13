import ctypes
from unittest.mock import patch

import pytest

from voicetotext.core.exceptions import OutputError
from voicetotext.output import win32_key_sender as wks
from voicetotext.output.key_sender import KeySender
from voicetotext.output.win32_key_sender import Win32KeySender


_VK_CONTROL = 0x11
_VK_V = 0x56
_KEYEVENTF_KEYUP = 0x0002


class TestWin32KeySenderSendCtrlV:
    @patch.object(wks, "_user32")
    def test_should_inject_ctrl_down_v_down_v_up_ctrl_up(self, mock_user32):
        mock_user32.SendInput.return_value = 4

        Win32KeySender().send_ctrl_v()

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
    def test_should_pass_the_real_input_struct_size(self, mock_user32):
        mock_user32.SendInput.return_value = 4

        Win32KeySender().send_ctrl_v()

        _, _, size = mock_user32.SendInput.call_args.args
        assert size == ctypes.sizeof(wks._Input)

    @patch.object(wks, "_user32")
    def test_should_raise_when_send_input_injects_nothing(self, mock_user32):
        mock_user32.SendInput.return_value = 0

        with pytest.raises(OutputError):
            Win32KeySender().send_ctrl_v()


class TestWin32KeySenderIsAKeySender:
    def test_should_satisfy_key_sender_protocol(self):
        assert isinstance(Win32KeySender(), KeySender)
