from unittest.mock import MagicMock

import pytest

from voicetotext.output.clipboard_write_error import ClipboardWriteError
from voicetotext.output.clipboard_writer import ClipboardWriter
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.key_sender import KeySender
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from voicetotext.output.sleeper import Sleeper


def _adapter() -> tuple[PasteAndClipboardAdapter, MagicMock, MagicMock, MagicMock]:
    sleeper = MagicMock(spec=Sleeper)
    writer = MagicMock(spec=ClipboardWriter)
    key_sender = MagicMock(spec=KeySender)
    adapter = PasteAndClipboardAdapter(
        sleeper=sleeper, writer=writer, key_sender=key_sender,
    )
    return adapter, sleeper, writer, key_sender


class TestPasteAndClipboardAdapterOutput:
    def test_should_write_to_clipboard_then_paste(self):
        adapter, _, writer, key_sender = _adapter()

        adapter.output("hello world")

        writer.set_text.assert_called_once_with("hello world")
        key_sender.send_ctrl_v.assert_called_once_with()

    def test_should_wait_before_pasting(self):
        adapter, sleeper, _, _ = _adapter()

        adapter.output("text")

        sleeper.sleep.assert_called_once()

    def test_should_log_warning_and_keep_text_when_paste_fails(self):
        adapter, _, writer, key_sender = _adapter()
        key_sender.send_ctrl_v.side_effect = RuntimeError("paste failed")

        adapter.output("hello world")

        writer.set_text.assert_called_once_with("hello world")

    def test_should_not_attempt_paste_when_clipboard_write_fails(self):
        adapter, _, writer, key_sender = _adapter()
        writer.set_text.side_effect = ClipboardWriteError("GlobalLock returned NULL")

        with pytest.raises(ClipboardWriteError):
            adapter.output("text")

        key_sender.send_ctrl_v.assert_not_called()


class TestPasteAndClipboardAdapterIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        adapter, *_ = _adapter()

        assert isinstance(adapter, TextOutputHandler)
