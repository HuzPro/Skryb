from unittest.mock import MagicMock

import pytest

from voicetotext.output.clipboard_write_error import ClipboardWriteError
from voicetotext.output.clipboard_writer import ClipboardWriter
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.key_sender import KeySender
from voicetotext.output.paste_and_clipboard_adapter import PasteAndClipboardAdapter
from voicetotext.output.paste_method import PasteMethod
from voicetotext.output.sleeper import Sleeper


def _adapter(
    paste_method=None,
) -> tuple[PasteAndClipboardAdapter, MagicMock, MagicMock, MagicMock]:
    sleeper = MagicMock(spec=Sleeper)
    writer = MagicMock(spec=ClipboardWriter)
    key_sender = MagicMock(spec=KeySender)
    adapter = PasteAndClipboardAdapter(
        sleeper=sleeper, writer=writer, key_sender=key_sender,
        paste_method=paste_method,
    )
    return adapter, sleeper, writer, key_sender


class TestPasteAndClipboardAdapterOutput:
    def test_should_write_to_clipboard_then_paste(self):
        adapter, _, writer, key_sender = _adapter()

        adapter.output("hello world")

        writer.set_text.assert_called_once_with("hello world")
        key_sender.send_paste.assert_called_once_with(PasteMethod.CTRL_V, "hello world")

    def test_pastes_with_the_method_chosen_for_the_app_in_front(self):
        adapter, _, _, key_sender = _adapter(lambda: PasteMethod.SHIFT_INSERT)

        adapter.output("ls -la")

        key_sender.send_paste.assert_called_once_with(PasteMethod.SHIFT_INSERT, "ls -la")

    def test_keeps_the_text_on_the_clipboard_when_the_app_cannot_be_identified(self):
        def unknown_app() -> PasteMethod:
            raise OSError("access denied")

        adapter, _, writer, key_sender = _adapter(unknown_app)

        adapter.output("hello world")

        writer.set_text.assert_called_once_with("hello world")
        key_sender.send_paste.assert_not_called()

    def test_should_wait_before_pasting(self):
        adapter, sleeper, _, _ = _adapter()

        adapter.output("text")

        sleeper.sleep.assert_called_once()

    def test_reports_the_paste_right_after_ctrl_v_goes_out(self):
        calls = MagicMock()
        writer = MagicMock(spec=ClipboardWriter)
        key_sender = MagicMock(spec=KeySender)
        calls.attach_mock(writer.set_text, "set_text")
        calls.attach_mock(key_sender.send_paste, "send_paste")
        adapter = PasteAndClipboardAdapter(
            sleeper=MagicMock(spec=Sleeper), writer=writer, key_sender=key_sender,
            on_pasted=calls.pasted,
        )

        adapter.output("ship it")

        assert [name for name, _args, _kwargs in calls.mock_calls] == [
            "set_text", "send_paste", "pasted",
        ]

    def test_still_reports_when_the_paste_could_not_be_sent(self):
        pasted = MagicMock()
        key_sender = MagicMock(spec=KeySender)
        key_sender.send_paste.side_effect = OSError("SendInput blocked")
        adapter = PasteAndClipboardAdapter(
            sleeper=MagicMock(spec=Sleeper), writer=MagicMock(spec=ClipboardWriter),
            key_sender=key_sender, on_pasted=pasted,
        )

        adapter.output("text stays on the clipboard")

        pasted.assert_called_once()

    def test_should_log_warning_and_keep_text_when_paste_fails(self):
        adapter, _, writer, key_sender = _adapter()
        key_sender.send_paste.side_effect = RuntimeError("paste failed")

        adapter.output("hello world")

        writer.set_text.assert_called_once_with("hello world")

    def test_should_not_attempt_paste_when_clipboard_write_fails(self):
        adapter, _, writer, key_sender = _adapter()
        writer.set_text.side_effect = ClipboardWriteError("GlobalLock returned NULL")

        with pytest.raises(ClipboardWriteError):
            adapter.output("text")

        key_sender.send_paste.assert_not_called()


class TestPasteAndClipboardAdapterIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        adapter, *_ = _adapter()

        assert isinstance(adapter, TextOutputHandler)
