import logging
from unittest.mock import MagicMock, call, patch

import pytest

from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.clipboard_preserving_decorator import (
    _PASTE_CONSUME_DELAY_SECONDS,
    ClipboardPreservingDecorator,
)
from voicetotext.output.handler import TextOutputHandler


_MODULE = "voicetotext.output.clipboard_preserving_decorator"


@pytest.fixture()
def log_capture(caplog):
    """Attach caplog handler to the module logger (propagate=False in this project)."""
    logger = logging.getLogger(_MODULE)
    logger.addHandler(caplog.handler)
    caplog.handler.setLevel(logging.DEBUG)
    yield caplog
    logger.removeHandler(caplog.handler)


def _build_decorator() -> tuple[
    ClipboardPreservingDecorator, MagicMock, MagicMock
]:
    inner = MagicMock(spec=TextOutputHandler)
    preserver = MagicMock(spec=ClipboardPreserver)
    decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)
    return decorator, inner, preserver


class TestClipboardPreservingDecoratorOutput:
    def test_should_delegate_to_inner_with_text(self):
        decorator, inner, _ = _build_decorator()

        decorator.output("hello world")

        inner.output.assert_called_once_with("hello world")

    def test_should_save_before_delegating(self):
        parent = MagicMock()
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        parent.attach_mock(inner, "inner")
        parent.attach_mock(preserver, "preserver")
        decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)

        decorator.output("text")

        save_index = parent.mock_calls.index(call.preserver.save())
        output_index = parent.mock_calls.index(call.inner.output("text"))
        assert save_index < output_index

    def test_should_restore_after_delegating(self):
        parent = MagicMock()
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        parent.attach_mock(inner, "inner")
        parent.attach_mock(preserver, "preserver")
        decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)

        decorator.output("text")

        output_index = parent.mock_calls.index(call.inner.output("text"))
        restore_index = parent.mock_calls.index(call.preserver.restore())
        assert output_index < restore_index

    def test_should_restore_even_when_inner_raises(self):
        decorator, inner, preserver = _build_decorator()
        inner.output.side_effect = RuntimeError("paste failed")

        with pytest.raises(RuntimeError):
            decorator.output("text")

        preserver.restore.assert_called_once()

    def test_should_skip_restore_when_save_fails(self, log_capture):
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        preserver.save.side_effect = OSError("clipboard locked")
        decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)

        decorator.output("text")

        preserver.restore.assert_not_called()
        assert "save failed" in log_capture.text.lower()

    def test_should_not_delegate_when_save_fails(self):
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        preserver.save.side_effect = OSError("clipboard locked")
        decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)

        decorator.output("text")

        inner.output.assert_not_called()

    def test_should_log_error_when_save_fails(self, log_capture):
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        preserver.save.side_effect = OSError("clipboard locked")
        decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)

        decorator.output("text")

        assert "snapshot failed" in log_capture.text.lower()
        error_records = [
            record for record in log_capture.records if record.levelname == "ERROR"
        ]
        assert len(error_records) >= 1

    def test_should_log_error_when_restore_fails(self, log_capture):
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        preserver.restore.side_effect = OSError("restore failed")
        decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)

        decorator.output("text")

        assert "restore" in log_capture.text.lower()

    @patch(f"{_MODULE}.time")
    def test_should_call_operations_in_correct_order_with_consume_delay(
        self, mock_time
    ):
        call_order: list[str] = []
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        preserver.save.side_effect = lambda: call_order.append("save")
        inner.output.side_effect = lambda t: call_order.append("inner.output")
        mock_time.sleep.side_effect = lambda d: call_order.append(f"sleep({d})")
        preserver.restore.side_effect = lambda: call_order.append("restore")
        decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)

        decorator.output("text")

        assert call_order == [
            "save",
            "inner.output",
            f"sleep({_PASTE_CONSUME_DELAY_SECONDS})",
            "restore",
        ]

    @patch(f"{_MODULE}.time")
    def test_should_skip_consume_delay_when_inner_raises(self, mock_time):
        call_order: list[str] = []
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        preserver.save.side_effect = lambda: call_order.append("save")
        inner.output.side_effect = RuntimeError("boom")
        mock_time.sleep.side_effect = lambda d: call_order.append("sleep")
        preserver.restore.side_effect = lambda: call_order.append("restore")
        decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)

        with pytest.raises(RuntimeError):
            decorator.output("text")

        assert call_order == ["save", "restore"]


class TestClipboardPreservingDecoratorIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        decorator = ClipboardPreservingDecorator(inner=inner, preserver=preserver)

        assert isinstance(decorator, TextOutputHandler)
