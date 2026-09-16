"""Tests for the UI Automation caret context reader.

The COM surface is hand-faked rather than mocked: a range is a document
string plus two offsets, which is exactly what ``IUIAutomationTextRange``
models, so the endpoint walk these tests exercise is the real one. Nothing
here imports ``comtypes``, so the file runs on any platform.
"""

from types import SimpleNamespace

import pytest

from voicetotext.output import win32_caret_context as module
from voicetotext.output.caret_context import NO_CONTEXT
from voicetotext.output.win32_caret_context import (
    CONTEXT_CHARS,
    _context_from,
    read_caret_context,
)


_TEXT_PATTERN_ID = 10014
_ENDPOINT_START = 0
_ENDPOINT_END = 1
_UNIT_CHARACTER = 0

_DOCUMENT = "Hello, I'm here for water. "
_SELECTION_START = 11
_SELECTION_END = 25


@pytest.fixture(autouse=True)
def stub_uia(monkeypatch) -> None:
    """Stand in for the generated ``UIAutomationClient`` module."""
    monkeypatch.setattr(
        module,
        "_uia",
        lambda: SimpleNamespace(
            UIA_TextPatternId=_TEXT_PATTERN_ID,
            IUIAutomationTextPattern=object(),
            TextPatternRangeEndpoint_Start=_ENDPOINT_START,
            TextPatternRangeEndpoint_End=_ENDPOINT_END,
            TextUnit_Character=_UNIT_CHARACTER,
        ),
    )


class NullPointer:
    """A null COM pointer, which is what comtypes hands back for no pattern."""

    def __bool__(self) -> bool:
        return False


class FakeTextRange:
    def __init__(self, document: str, start: int, end: int) -> None:
        self.document = document
        self.start = start
        self.end = end

    def Clone(self) -> "FakeTextRange":  # noqa: N802 (COM name)
        return FakeTextRange(self.document, self.start, self.end)

    def GetText(self, max_length: int) -> str:  # noqa: N802 (COM name)
        return self.document[self.start:self.end]

    def MoveEndpointByRange(  # noqa: N802 (COM name)
        self, endpoint: int, target: "FakeTextRange", target_endpoint: int,
    ) -> None:
        self._move_to(
            endpoint,
            target.start if target_endpoint == _ENDPOINT_START else target.end,
        )

    def MoveEndpointByUnit(  # noqa: N802 (COM name)
        self, endpoint: int, unit: int, count: int,
    ) -> int:
        assert unit == _UNIT_CHARACTER
        current = self.start if endpoint == _ENDPOINT_START else self.end
        self._move_to(endpoint, max(0, min(len(self.document), current + count)))
        return count

    def _move_to(self, endpoint: int, position: int) -> None:
        if endpoint == _ENDPOINT_START:
            self.start = position
        else:
            self.end = position


class FakeTextRangeArray:
    def __init__(self, ranges: list[FakeTextRange]) -> None:
        self.ranges = ranges

    @property
    def Length(self) -> int:  # noqa: N802 (COM name)
        return len(self.ranges)

    def GetElement(self, index: int) -> FakeTextRange:  # noqa: N802 (COM name)
        return self.ranges[index]


class FakeTextPattern:
    def __init__(self, ranges: list[FakeTextRange]) -> None:
        self.ranges = ranges

    def QueryInterface(self, interface: object) -> "FakeTextPattern":  # noqa: N802 (COM name)
        return self

    def GetSelection(self) -> FakeTextRangeArray:  # noqa: N802 (COM name)
        return FakeTextRangeArray(self.ranges)


class FakeElement:
    def __init__(self, pattern: object) -> None:
        self.pattern = pattern

    def GetCurrentPattern(self, pattern_id: int) -> object:  # noqa: N802 (COM name)
        assert pattern_id == _TEXT_PATTERN_ID
        return self.pattern


class FakeAutomation:
    def __init__(self, element: FakeElement) -> None:
        self.element = element

    def GetFocusedElement(self) -> FakeElement:  # noqa: N802 (COM name)
        return self.element


class BrokenAutomation:
    def GetFocusedElement(self) -> FakeElement:  # noqa: N802 (COM name)
        raise OSError("the UI Automation client went away")


def _automation_over(document: str, start: int, end: int) -> FakeAutomation:
    selection = FakeTextRange(document, start, end)
    return FakeAutomation(FakeElement(FakeTextPattern([selection])))


class TestContextFromASelection:
    def test_should_split_the_document_around_the_selection(self):
        automation = _automation_over(_DOCUMENT, _SELECTION_START, _SELECTION_END)

        context = _context_from(automation)

        assert context.before == "Hello, I'm "
        assert context.selected == "here for water"
        assert context.after == ". "

    def test_should_report_an_empty_selection_for_a_bare_caret(self):
        automation = _automation_over(_DOCUMENT, _SELECTION_START, _SELECTION_START)

        context = _context_from(automation)

        assert context.selected == ""
        assert context.before == "Hello, I'm "
        assert context.after == "here for water. "

    def test_should_read_no_more_than_the_context_width_on_each_side(self):
        document = "a" * 200 + "SELECTED" + "b" * 200
        automation = _automation_over(document, 200, 208)

        context = _context_from(automation)

        assert context.before == "a" * CONTEXT_CHARS
        assert context.after == "b" * CONTEXT_CHARS


class TestContextWhenNoneCanBeRead:
    def test_should_report_no_context_when_the_element_has_no_text_pattern(self):
        automation = FakeAutomation(FakeElement(NullPointer()))

        assert _context_from(automation) == NO_CONTEXT

    def test_should_report_no_context_when_nothing_is_selected(self):
        automation = FakeAutomation(FakeElement(FakeTextPattern([])))

        assert _context_from(automation) == NO_CONTEXT

    def test_should_report_no_context_when_ui_automation_fails(self, monkeypatch):
        monkeypatch.setattr(module, "_automation", BrokenAutomation)

        assert read_caret_context() == NO_CONTEXT

    def test_should_report_no_context_when_the_automation_client_cannot_be_built(
        self, monkeypatch,
    ):
        def explode() -> FakeAutomation:
            raise OSError("UIAutomationCore.dll is missing")

        monkeypatch.setattr(module, "_automation", explode)

        assert read_caret_context() == NO_CONTEXT


class TestReadCaretContext:
    def test_should_read_the_focused_element(self, monkeypatch):
        automation = _automation_over(_DOCUMENT, _SELECTION_START, _SELECTION_END)
        monkeypatch.setattr(module, "_automation", lambda: automation)

        assert read_caret_context().selected == "here for water"
