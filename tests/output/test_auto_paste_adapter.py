from unittest.mock import call, patch

from voicetotext.output.auto_paste_adapter import AutoPasteAdapter
from voicetotext.output.handler import TextOutputHandler


class TestAutoPasteAdapterOutput:
    @patch("voicetotext.output.auto_paste_adapter.time")
    @patch("voicetotext.output.auto_paste_adapter.pyautogui")
    @patch("voicetotext.output.auto_paste_adapter.pyperclip")
    def test_should_copy_to_clipboard_then_paste(
        self, mock_pyperclip, mock_pyautogui, mock_time
    ):
        adapter = AutoPasteAdapter()

        adapter.output("hello world")

        mock_pyperclip.copy.assert_called_once_with("hello world")
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")

    @patch("voicetotext.output.auto_paste_adapter.time")
    @patch("voicetotext.output.auto_paste_adapter.pyautogui")
    @patch("voicetotext.output.auto_paste_adapter.pyperclip")
    def test_should_wait_before_pasting(
        self, mock_pyperclip, mock_pyautogui, mock_time
    ):
        adapter = AutoPasteAdapter()

        adapter.output("text")

        mock_time.sleep.assert_called_once_with(0.1)


class TestAutoPasteAdapterIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        adapter = AutoPasteAdapter()

        assert isinstance(adapter, TextOutputHandler)
