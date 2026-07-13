import logging
from unittest.mock import MagicMock, call

import pytest

from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.clipboard_preserving_decorator import (
    ClipboardPreservingDecorator,
)
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.sleeper import Sleeper


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
    ClipboardPreservingDecorator, MagicMock, MagicMock, MagicMock
]:
    inner = MagicMock(spec=TextOutputHandler)
    preserver = MagicMock(spec=ClipboardPreserver)
    sleeper = MagicMock(spec=Sleeper)
    decorator = ClipboardPreservingDecorator(
        inner=inner, preserver=preserver, sleeper=sleeper
    )
    return decorator, inner, preserver, sleeper


class TestClipboardPreservingDecoratorOutput:
    def test_should_delegate_to_inner_with_text(self):
        decorator, inner, _, _ = _build_decorator()

        decorator.output("hello world")

        inner.output.assert_called_once_with("hello world")

    def test_should_save_before_delegating(self):
        parent = MagicMock()
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        sleeper = MagicMock(spec=Sleeper)
        parent.attach_mock(inner, "inner")
        parent.attach_mock(preserver, "preserver")
        decorator = ClipboardPreservingDecorator(
            inner=inner, preserver=preserver, sleeper=sleeper
        )

        decorator.output("text")

        save_index = parent.mock_calls.index(call.preserver.save())
        output_index = parent.mock_calls.index(call.inner.output("text"))
        assert save_index < output_index

    def test_should_restore_after_delegating(self):
        parent = MagicMock()
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        sleeper = MagicMock(spec=Sleeper)
        parent.attach_mock(inner, "inner")
        parent.attach_mock(preserver, "preserver")
        decorator = ClipboardPreservingDecorator(
            inner=inner, preserver=preserver, sleeper=sleeper
        )

        decorator.output("text")

        output_index = parent.mock_calls.index(call.inner.output("text"))
        restore_index = parent.mock_calls.index(call.preserver.restore())
        assert output_index < restore_index

    def test_should_restore_even_when_inner_raises(self):
        decorator, inner, preserver, _ = _build_decorator()
        inner.output.side_effect = RuntimeError("paste failed")

        with pytest.raises(RuntimeError):
            decorator.output("text")

        preserver.restore.assert_called_once()

    def test_should_skip_restore_when_save_fails(self, log_capture):
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        preserver.save.side_effect = OSError("clipboard locked")
        sleeper = MagicMock(spec=Sleeper)
        decorator = ClipboardPreservingDecorator(
            inner=inner, preserver=preserver, sleeper=sleeper
        )

        decorator.output("text")

        preserver.restore.assert_not_called()
        assert "save failed" in log_capture.text.lower()

    def test_should_not_delegate_when_save_fails(self):
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        preserver.save.side_effect = OSError("clipboard locked")
        sleeper = MagicMock(spec=Sleeper)
        decorator = ClipboardPreservingDecorator(
            inner=inner, preserver=preserver, sleeper=sleeper
        )

        decorator.output("text")

        inner.output.assert_not_called()

    def test_should_log_error_when_save_fails(self, log_capture):
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        preserver.save.side_effect = OSError("clipboard locked")
        sleeper = MagicMock(spec=Sleeper)
        decorator = ClipboardPreservingDecorator(
            inner=inner, preserver=preserver, sleeper=sleeper
        )

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
        sleeper = MagicMock(spec=Sleeper)
        decorator = ClipboardPreservingDecorator(
            inner=inner, preserver=preserver, sleeper=sleeper
        )

        decorator.output("text")

        assert "restore" in log_capture.text.lower()

    def test_should_sleep_between_paste_and_restore(self):
        call_order: list[str] = []
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        sleeper = MagicMock(spec=Sleeper)
        preserver.save.side_effect = lambda: call_order.append("save")
        inner.output.side_effect = lambda t: call_order.append("inner.output")
        sleeper.sleep.side_effect = lambda d: call_order.append("sleep")
        preserver.restore.side_effect = lambda: call_order.append("restore")
        decorator = ClipboardPreservingDecorator(
            inner=inner, preserver=preserver, sleeper=sleeper
        )

        decorator.output("text")

        assert call_order == ["save", "inner.output", "sleep", "restore"]

    def test_should_skip_consume_delay_when_inner_raises(self):
        call_order: list[str] = []
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        sleeper = MagicMock(spec=Sleeper)
        preserver.save.side_effect = lambda: call_order.append("save")
        inner.output.side_effect = RuntimeError("boom")
        sleeper.sleep.side_effect = lambda d: call_order.append("sleep")
        preserver.restore.side_effect = lambda: call_order.append("restore")
        decorator = ClipboardPreservingDecorator(
            inner=inner, preserver=preserver, sleeper=sleeper
        )

        with pytest.raises(RuntimeError):
            decorator.output("text")

        assert call_order == ["save", "restore"]

    def test_should_give_target_at_least_a_quarter_second_to_consume_the_paste(self):
        # The target app reads the clipboard asynchronously after the
        # injected Ctrl+V. Restoring the snapshot too early makes slow
        # targets paste the OLD clipboard contents instead of the
        # transcription. 50ms was observed losing that race in the wild.
        decorator, _, _, sleeper = _build_decorator()

        decorator.output("text")

        (delay,), _ = sleeper.sleep.call_args
        assert delay >= 0.25

    def test_should_delegate_sleep_to_injected_sleeper(self):
        decorator, _, _, sleeper = _build_decorator()

        decorator.output("text")

        sleeper.sleep.assert_called_once()


class TestClipboardPreservingDecoratorIsATextOutputHandler:
    def test_should_implement_text_output_handler_interface(self):
        inner = MagicMock(spec=TextOutputHandler)
        preserver = MagicMock(spec=ClipboardPreserver)
        sleeper = MagicMock(spec=Sleeper)
        decorator = ClipboardPreservingDecorator(
            inner=inner, preserver=preserver, sleeper=sleeper
        )

        assert isinstance(decorator, TextOutputHandler)
