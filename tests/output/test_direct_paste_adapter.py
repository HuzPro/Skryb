import logging
from unittest.mock import MagicMock, call, patch

import pytest

from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.direct_paste_adapter import DirectPasteAdapter
from voicetotext.output.handler import TextOutputHandler


_MODULE = "voicetotext.output.direct_paste_adapter"


@pytest.fixture()
def log_capture(caplog):
    """Attach caplog handler to the module logger (propagate=False in this project)."""
    logger = logging.getLogger(_MODULE)
    logger.addHandler(caplog.handler)
    caplog.handler.setLevel(logging.DEBUG)
    yield caplog
    logger.removeHandler(caplog.handler)


def _build_adapter(
    preserver: ClipboardPreserver | None = None,
) -> tuple[DirectPasteAdapter, MagicMock]:
    mock_preserver = preserver or MagicMock(spec=ClipboardPreserver)
    adapter = DirectPasteAdapter(mock_preserver)
    return adapter, mock_preserver


class TestDirectPasteAdapterOutput:
    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    @patch(f"{_MODULE}.time")
    def test_should_copy_text_to_clipboard(
        self, mock_time, mock_pyperclip, mock_pyautogui
    ):
        adapter, _ = _build_adapter()

        adapter.output("hello world")

        mock_pyperclip.copy.assert_called_once_with("hello world")

    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    @patch(f"{_MODULE}.time")
    def test_should_simulate_ctrl_v(
        self, mock_time, mock_pyperclip, mock_pyautogui
    ):
        adapter, _ = _build_adapter()

        adapter.output("text")

        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")

    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    @patch(f"{_MODULE}.time")
    def test_should_restore_clipboard_after_paste(
        self, mock_time, mock_pyperclip, mock_pyautogui
    ):
        adapter, mock_preserver = _build_adapter()

        adapter.output("text")

        mock_preserver.restore.assert_called_once()

    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    @patch(f"{_MODULE}.time")
    def test_should_restore_clipboard_even_when_paste_fails(
        self, mock_time, mock_pyperclip, mock_pyautogui
    ):
        mock_pyautogui.hotkey.side_effect = RuntimeError("paste failed")
        adapter, mock_preserver = _build_adapter()

        adapter.output("text")

        mock_preserver.restore.assert_called_once()

    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    @patch(f"{_MODULE}.time")
    def test_should_skip_restore_when_save_fails(
        self, mock_time, mock_pyperclip, mock_pyautogui, log_capture
    ):
        mock_preserver = MagicMock(spec=ClipboardPreserver)
        mock_preserver.save.side_effect = OSError("clipboard locked")
        adapter = DirectPasteAdapter(mock_preserver)

        adapter.output("text")

        mock_preserver.restore.assert_not_called()
        assert "save failed" in log_capture.text.lower()

    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    @patch(f"{_MODULE}.time")
    def test_should_still_paste_when_save_fails(
        self, mock_time, mock_pyperclip, mock_pyautogui
    ):
        mock_preserver = MagicMock(spec=ClipboardPreserver)
        mock_preserver.save.side_effect = OSError("clipboard locked")
        adapter = DirectPasteAdapter(mock_preserver)

        adapter.output("text")

        mock_pyperclip.copy.assert_called_once_with("text")
        mock_pyautogui.hotkey.assert_called_once_with("ctrl", "v")

    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    @patch(f"{_MODULE}.time")
    def test_should_log_error_when_restore_fails(
        self, mock_time, mock_pyperclip, mock_pyautogui, log_capture
    ):
        mock_preserver = MagicMock(spec=ClipboardPreserver)
        mock_preserver.restore.side_effect = OSError("restore failed")
        adapter = DirectPasteAdapter(mock_preserver)

        adapter.output("text")

        assert "restore" in log_capture.text.lower()

    @patch(f"{_MODULE}.pyautogui")
    @patch(f"{_MODULE}.pyperclip")
    @patch(f"{_MODULE}.time")
    def test_should_call_operations_in_correct_order_with_delays(
        self, mock_time, mock_pyperclip, mock_pyautogui
    ):
        call_order: list[str] = []
        mock_preserver = MagicMock(spec=ClipboardPreserver)
        mock_preserver.save.side_effect = lambda: call_order.append("save")
        mock_pyperclip.copy.side_effect = lambda t: call_order.append("copy")
        mock_time.sleep.side_effect = lambda d: call_order.append(f"sleep({d})")
        mock_pyautogui.hotkey.side_effect = lambda *a: call_order.append("hotkey")
        mock_preserver.restore.side_effect = lambda: call_order.append("restore")

        adapter = DirectPasteAdapter(mock_preserver)
        adapter.output("text")

        assert call_order == [
            "save",
            "copy",
            "sleep(0.1)",
            "hotkey",
            "sleep(0.05)",
            "restore",
        ]


class TestDirectPasteAdapterIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        mock_preserver = MagicMock(spec=ClipboardPreserver)
        adapter = DirectPasteAdapter(mock_preserver)

        assert isinstance(adapter, TextOutputHandler)
