from unittest.mock import MagicMock

from voicetotext.output.clipboard_adapter import ClipboardAdapter
from voicetotext.output.clipboard_writer import ClipboardWriter
from voicetotext.output.handler import TextOutputHandler


class TestClipboardAdapterOutput:
    def test_should_write_text_to_clipboard(self):
        writer = MagicMock(spec=ClipboardWriter)
        adapter = ClipboardAdapter(writer=writer)

        adapter.output("hello world")

        writer.set_text.assert_called_once_with("hello world")


class TestClipboardAdapterIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        writer = MagicMock(spec=ClipboardWriter)
        adapter = ClipboardAdapter(writer=writer)

        assert isinstance(adapter, TextOutputHandler)
