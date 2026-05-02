from unittest.mock import MagicMock, patch

import pytest

from voicetotext.output.clipboard_write_error import ClipboardWriteError
from voicetotext.output.clipboard_writer import ClipboardWriter
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from voicetotext.output.sleeper import Sleeper


_MODULE = "voicetotext.output.paste_and_clipboard_adapter"


class TestPasteAndClipboardAdapterOutput:
    @patch(f"{_MODULE}.pyautogui")
    def test_should_write_to_clipboard_then_paste(self, mock_pyautogui):
        sleeper = MagicMock(spec=Sleeper)
        writer = MagicMock(spec=ClipboardWriter)
        adapter = PasteAndClipboardAdapter(sleeper=sleeper, writer=writer)

        adapter.output("hello world")

        writer.set_text.assert_called_once_with("hello world")
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")

    @patch(f"{_MODULE}.pyautogui")
    def test_should_wait_before_pasting(self, mock_pyautogui):
        sleeper = MagicMock(spec=Sleeper)
        writer = MagicMock(spec=ClipboardWriter)
        adapter = PasteAndClipboardAdapter(sleeper=sleeper, writer=writer)

        adapter.output("text")

        sleeper.sleep.assert_called_once()

    @patch(f"{_MODULE}.pyautogui")
    def test_should_log_warning_and_keep_text_when_paste_fails(
        self, mock_pyautogui
    ):
        mock_pyautogui.hotkey.side_effect = RuntimeError("paste failed")
        sleeper = MagicMock(spec=Sleeper)
        writer = MagicMock(spec=ClipboardWriter)
        adapter = PasteAndClipboardAdapter(sleeper=sleeper, writer=writer)

        adapter.output("hello world")

        writer.set_text.assert_called_once_with("hello world")

    @patch(f"{_MODULE}.pyautogui")
    def test_should_not_attempt_paste_when_clipboard_write_fails(
        self, mock_pyautogui
    ):
        sleeper = MagicMock(spec=Sleeper)
        writer = MagicMock(spec=ClipboardWriter)
        writer.set_text.side_effect = ClipboardWriteError("GlobalLock returned NULL")
        adapter = PasteAndClipboardAdapter(sleeper=sleeper, writer=writer)

        with pytest.raises(ClipboardWriteError):
            adapter.output("text")

        mock_pyautogui.hotkey.assert_not_called()


class TestPasteAndClipboardAdapterIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        sleeper = MagicMock(spec=Sleeper)
        writer = MagicMock(spec=ClipboardWriter)
        adapter = PasteAndClipboardAdapter(sleeper=sleeper, writer=writer)

        assert isinstance(adapter, TextOutputHandler)
