from unittest.mock import patch

from voicetotext.output.clipboard_adapter import ClipboardAdapter
from voicetotext.output.handler import TextOutputHandler


class TestClipboardAdapterOutput:
    @patch("voicetotext.output.clipboard_adapter.pyperclip")
    def test_should_copy_text_to_clipboard(self, mock_pyperclip):
        adapter = ClipboardAdapter()

        adapter.output("hello world")

        mock_pyperclip.copy.assert_called_once_with("hello world")


class TestClipboardAdapterIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        adapter = ClipboardAdapter()

        assert isinstance(adapter, TextOutputHandler)
