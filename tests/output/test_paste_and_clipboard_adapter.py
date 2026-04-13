from unittest.mock import patch

from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter


_MODULE = "voicetotext.output.paste_and_clipboard_adapter"


class TestPasteAndClipboardAdapterOutput:
    @patch(f"{_MODULE}.time")
    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    def test_should_copy_to_clipboard_then_paste(
        self, mock_pyperclip, mock_pyautogui, mock_time
    ):
        adapter = PasteAndClipboardAdapter()

        adapter.output("hello world")

        mock_pyperclip.copy.assert_called_once_with("hello world")
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")

    @patch(f"{_MODULE}.time")
    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    def test_should_wait_before_pasting(
        self, mock_pyperclip, mock_pyautogui, mock_time
    ):
        adapter = PasteAndClipboardAdapter()

        adapter.output("text")

        mock_time.sleep.assert_called_once_with(0.1)


    @patch(f"{_MODULE}.time")
    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    def test_should_keep_text_in_clipboard_when_paste_fails(
        self, mock_pyperclip, mock_pyautogui, mock_time
    ):
        mock_pyautogui.hotkey.side_effect = RuntimeError("paste failed")
        adapter = PasteAndClipboardAdapter()

        adapter.output("hello world")

        mock_pyperclip.copy.assert_called_once_with("hello world")


class TestPasteAndClipboardAdapterIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        adapter = PasteAndClipboardAdapter()

        assert isinstance(adapter, TextOutputHandler)
