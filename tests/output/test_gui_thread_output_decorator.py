"""Tests for :class:`GuiThreadOutputDecorator`.

The decorator's only job is to forward ``output(text)`` to its inner
handler via a :class:`GuiThreadDispatcher`. Threading is a detail of
the dispatcher; the decorator itself is plain Python and can be tested
without Qt.
"""

from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.output.gui_thread_output_decorator import (
    GuiThreadOutputDecorator,
)
from voicetotext.output.handler import TextOutputHandler


class _RecordingInner(TextOutputHandler):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def output(self, text: str) -> None:
        self.calls.append(text)


class _ImmediateDispatcher(GuiThreadDispatcher):
    """Test double that runs actions inline on the calling thread."""

    def dispatch(self, action) -> None:
        action()


class _DeferredDispatcher(GuiThreadDispatcher):
    """Test double that captures actions for the test to invoke later."""

    def __init__(self) -> None:
        self.queued: list = []

    def dispatch(self, action) -> None:
        self.queued.append(action)


class TestGuiThreadOutputDecorator:
    def test_should_implement_text_output_handler(self) -> None:
        decorator = GuiThreadOutputDecorator(
            inner=_RecordingInner(), dispatcher=_ImmediateDispatcher(),
        )

        assert isinstance(decorator, TextOutputHandler)

    def test_should_forward_text_to_inner_via_dispatcher(self) -> None:
        inner = _RecordingInner()
        decorator = GuiThreadOutputDecorator(
            inner=inner, dispatcher=_ImmediateDispatcher(),
        )

        decorator.output("hello world")

        assert inner.calls == ["hello world"]

    def test_should_not_call_inner_directly_on_calling_thread(self) -> None:
        # If the dispatcher defers the action, the inner must not have
        # been called yet — proving the decorator hands off via the
        # dispatcher rather than invoking inline.
        inner = _RecordingInner()
        dispatcher = _DeferredDispatcher()
        decorator = GuiThreadOutputDecorator(inner=inner, dispatcher=dispatcher)

        decorator.output("queued")

        assert inner.calls == []
        assert len(dispatcher.queued) == 1

        dispatcher.queued[0]()
        assert inner.calls == ["queued"]

    def test_should_dispatch_each_call_separately(self) -> None:
        inner = _RecordingInner()
        dispatcher = _DeferredDispatcher()
        decorator = GuiThreadOutputDecorator(inner=inner, dispatcher=dispatcher)

        decorator.output("first")
        decorator.output("second")

        assert len(dispatcher.queued) == 2
        for action in dispatcher.queued:
            action()
        assert inner.calls == ["first", "second"]
