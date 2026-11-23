"""Tests for the UI Automation caret context reader.

The COM surface is hand-faked rather than mocked: a range is a document
string plus two offsets, which is exactly what ``IUIAutomationTextRange``
models, so the endpoint walk these tests exercise is the real one. A page
and the editable field inside it are two ranges over the same string, the
way Chromium reports them. Nothing here imports ``comtypes``, so the file
runs on any platform.
"""

import functools
import logging
from types import SimpleNamespace

import pytest

from tests.fakes.config import FakeConfig
from skryb.diagnostics.session import content_recording_enabled
from skryb.output import win32_caret_context as module
from skryb.output.caret_context import (
    CONTEXT_CHARS,
    NO_CONTEXT,
    SELECTION_OUTSIDE_FIELD,
    CaretContext,
    fit_to_context,
    log_caret_text_when,
)
from skryb.output.win32_caret_context import (
    _context_from,
    read_caret_context,
)


_TEXT_PATTERN_ID = 10014
_EDIT = 50004
_PANE = 50033
_HAS_KEYBOARD_FOCUS = 30008
_DESCENDANTS = 4
_ENDPOINT_START = 0
_ENDPOINT_END = 1
_UNIT_CHARACTER = 0
_UNIT_LINE = 3

_DOCUMENT = "Hello, I'm here for water. "
_SELECTION_START = 11
_SELECTION_END = 25

_CHROME_BEFORE = "png\nRemove image.png\n"
_CHROME_AFTER = "\nSend\nAdd\nPress an"


@pytest.fixture(autouse=True)
def stub_uia(monkeypatch) -> None:
    """Stand in for the generated ``UIAutomationClient`` module."""
    monkeypatch.setattr(
        module,
        "_uia",
        lambda: SimpleNamespace(
            UIA_TextPatternId=_TEXT_PATTERN_ID,
            UIA_PaneControlTypeId=_PANE,
            UIA_HasKeyboardFocusPropertyId=_HAS_KEYBOARD_FOCUS,
            TreeScope_Descendants=_DESCENDANTS,
            IUIAutomationTextPattern=object(),
            TextPatternRangeEndpoint_Start=_ENDPOINT_START,
            TextPatternRangeEndpoint_End=_ENDPOINT_END,
            TextUnit_Character=_UNIT_CHARACTER,
            TextUnit_Line=_UNIT_LINE,
        ),
    )


@pytest.fixture(autouse=True)
def silent_edit_control_reader(monkeypatch) -> None:
    """Keep the fallback from reading the real desktop during these tests."""
    monkeypatch.setattr(module, "read_edit_control_context", lambda: NO_CONTEXT)


class NullPointer:
    """A null COM pointer, which is what comtypes hands back for no pattern."""

    def __bool__(self) -> bool:
        return False


class FakeTextRange:
    """A range over a flat text. ``past_the_text`` makes it compare as that
    many positions later than its place in the text, the way Chromium orders
    a caret at the end of a composer after the composer's own range.
    Lines start after each line break and at ``line_starts``, where Chromium
    starts a paragraph or wraps a line with no character in between."""

    def __init__(
        self, document: str, start: int, end: int, past_the_text: int = 0,
        line_starts: frozenset[int] = frozenset(),
    ) -> None:
        self.document = document
        self.start = start
        self.end = end
        self.past_the_text = past_the_text
        self.line_starts = line_starts

    def Clone(self) -> "FakeTextRange":  # noqa: N802 (COM name)
        return FakeTextRange(
            self.document, self.start, self.end, self.past_the_text, self.line_starts,
        )

    def ExpandToEnclosingUnit(self, unit: int) -> None:
        assert unit == _UNIT_LINE
        starts = self.line_starts | {0} | {
            index + 1 for index, character in enumerate(self.document) if character in "\r\n"
        }
        self.start = max(position for position in starts if position <= self.start)
        self.end = min((position for position in starts if position > self.start), default=len(self.document))

    def GetText(self, max_length: int) -> str:  # noqa: N802 (COM name)
        return self.document[self.start:self.end]

    def MoveEndpointByRange(  # noqa: N802 (COM name)
        self, endpoint: int, target: "FakeTextRange", target_endpoint: int,
    ) -> None:
        self._move_to(endpoint, target.position_of(target_endpoint))

    def CompareEndpoints(  # noqa: N802 (COM name)
        self, endpoint: int, target: "FakeTextRange", target_endpoint: int,
    ) -> int:
        return (
            self.position_of(endpoint) + self.past_the_text
            - target.position_of(target_endpoint) - target.past_the_text
        )

    def position_of(self, endpoint: int) -> int:
        return self.start if endpoint == _ENDPOINT_START else self.end

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
    def __init__(self, ranges: list[FakeTextRange], document: FakeTextRange) -> None:
        self.ranges = ranges
        self.document = document

    @property
    def DocumentRange(self) -> FakeTextRange:  # noqa: N802 (COM name)
        return self.document

    def QueryInterface(self, interface: object) -> "FakeTextPattern":  # noqa: N802 (COM name)
        return self

    def GetSelection(self) -> FakeTextRangeArray:  # noqa: N802 (COM name)
        return FakeTextRangeArray(self.ranges)


class FakeElement:
    def __init__(
        self,
        pattern: object,
        is_password: bool = False,
        *,
        control_type: int = _EDIT,
        class_name: str = "",
        framework: str = "Chrome",
        focused_inside: "FakeElement | None" = None,
    ) -> None:
        self.pattern = pattern
        self.CurrentIsPassword = int(is_password)  # noqa: N815 (COM name)
        self.CurrentControlType = control_type  # noqa: N815 (COM name)
        self.CurrentClassName = class_name  # noqa: N815 (COM name)
        self.CurrentFrameworkId = framework  # noqa: N815 (COM name)
        self.CurrentNativeWindowHandle = None  # noqa: N815 (COM name)
        self.focused_inside = focused_inside

    def GetCurrentPattern(self, pattern_id: int) -> object:  # noqa: N802 (COM name)
        assert pattern_id == _TEXT_PATTERN_ID
        return self.pattern

    def FindFirst(self, scope: int, condition: object) -> "FakeElement | None":  # noqa: N802 (COM name)
        assert scope == _DESCENDANTS
        assert condition == (_HAS_KEYBOARD_FOCUS, True)
        return self.focused_inside


class FakeAutomation:
    def __init__(self, element: FakeElement) -> None:
        self.element = element

    def GetFocusedElement(self) -> FakeElement:  # noqa: N802 (COM name)
        return self.element

    def CreatePropertyCondition(self, property_id: int, value: object) -> tuple:  # noqa: N802 (COM name)
        return (property_id, value)


class BrokenAutomation:
    def GetFocusedElement(self) -> FakeElement:  # noqa: N802 (COM name)
        raise OSError("the UI Automation client went away")


def _whole_of(document: str) -> FakeTextRange:
    return FakeTextRange(document, 0, len(document))


def _automation_over(document: str, start: int, end: int) -> FakeAutomation:
    return _automation_within(_whole_of(document), start, end)


def _automation_within(field: FakeTextRange, start: int, end: int) -> FakeAutomation:
    selection = FakeTextRange(field.document, start, end)
    return FakeAutomation(FakeElement(FakeTextPattern([selection], field)))


def _automation_in_a_page(field_text: str, caret: int) -> FakeAutomation:
    """A caret in an editable field with page chrome on both sides of it."""
    page = _CHROME_BEFORE + field_text + _CHROME_AFTER
    start = len(_CHROME_BEFORE)
    field = FakeTextRange(page, start, start + len(field_text))
    position = start + caret
    return _automation_within(field, position, position)


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

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (0, 0, CaretContext("", "", _DOCUMENT)),
            (27, 27, CaretContext(_DOCUMENT, "", "")),
            (25, 27, CaretContext("Hello, I'm here for water", ". ", "")),
            (0, 27, CaretContext("", _DOCUMENT, "")),
        ],
        ids=["caret at the start", "caret at the end", "selection to the end", "everything"],
    )
    def test_should_read_up_to_the_edges_of_the_document(self, start, end, expected):
        automation = _automation_over(_DOCUMENT, start, end)

        assert _context_from(automation) == expected


class TestFieldsInsideALargerDocument:
    """Chromium reports the page and the composer as one flat text tree."""

    def test_should_not_read_the_page_before_the_field(self):
        automation = _automation_in_a_page("Hello there. ", caret=0)

        assert _context_from(automation).before == ""

    def test_should_not_read_the_page_after_the_field(self):
        field = "Hello there. "
        automation = _automation_in_a_page(field, caret=len(field))

        assert _context_from(automation).after == ""

    def test_should_report_no_context_when_the_selection_is_outside_the_field(self):
        page = _CHROME_BEFORE + "Hello there. " + _CHROME_AFTER
        field = FakeTextRange(page, len(_CHROME_BEFORE), len(page) - len(_CHROME_AFTER))
        automation = _automation_within(field, 0, 0)

        assert _context_from(automation) is SELECTION_OUTSIDE_FIELD

    def test_a_selection_running_past_the_end_of_the_field_is_read_up_to_its_edge(self):
        text = "I went to the store and bought milk. "
        page = _CHROME_BEFORE + text + _CHROME_AFTER
        start = len(_CHROME_BEFORE)
        field = FakeTextRange(page, start, start + len(text) - 1)
        selected_from = start + text.index("bought")
        automation = _automation_within(field, selected_from, start + len(text) + 1)

        context = _context_from(automation)

        assert context == CaretContext("I went to the store and ", "bought milk.", "")

    def test_a_selection_starting_before_the_field_is_read_from_its_edge(self):
        text = "bought milk. Then home."
        page = _CHROME_BEFORE + text + _CHROME_AFTER
        start = len(_CHROME_BEFORE)
        field = FakeTextRange(page, start, start + len(text))
        automation = _automation_within(field, start - 3, start + len("bought milk. "))

        context = _context_from(automation)

        assert context == CaretContext("", "bought milk. ", "Then home.")

    def test_a_selection_wholly_after_the_field_is_still_outside_it(self):
        text = "Hello there. "
        page = _CHROME_BEFORE + text + _CHROME_AFTER
        start = len(_CHROME_BEFORE)
        field = FakeTextRange(page, start, start + len(text))
        after_field = start + len(text)
        automation = _automation_within(field, after_field + 1, after_field + 4)

        assert _context_from(automation) is SELECTION_OUTSIDE_FIELD

    def test_a_bare_caret_after_the_field_with_text_in_between_is_still_outside_it(self):
        text = "Hello there. "
        page = _CHROME_BEFORE + text + _CHROME_AFTER
        start = len(_CHROME_BEFORE)
        field = FakeTextRange(page, start, start + len(text))
        automation = _automation_within(field, start + len(text) + 1, start + len(text) + 1)

        assert _context_from(automation) is SELECTION_OUTSIDE_FIELD

    @pytest.mark.parametrize(
        "text",
        ["Pack the lantern, and ", "No space at the end", "x", "naïve الطرد. " * 40],
        ids=["trailing space", "no trailing space", "one character", "long, with unicode"],
    )
    def test_a_caret_reported_just_past_the_end_of_the_text_is_read_as_the_end_of_the_field(self, text):
        page = _CHROME_BEFORE + text + _CHROME_AFTER
        start = len(_CHROME_BEFORE)
        field = FakeTextRange(page, start, start + len(text))
        caret = FakeTextRange(page, start + len(text), start + len(text), past_the_text=1)
        automation = FakeAutomation(FakeElement(FakeTextPattern([caret], field)))

        context = _context_from(automation)

        assert context == CaretContext(text[-CONTEXT_CHARS:], "", "")
        assert context is not SELECTION_OUTSIDE_FIELD

    def test_a_second_dictation_carries_on_the_sentence_the_first_left_unfinished(self):
        text = "Pack the lantern, and "
        page = _CHROME_BEFORE + text + _CHROME_AFTER
        start = len(_CHROME_BEFORE)
        field = FakeTextRange(page, start, start + len(text))
        caret = FakeTextRange(page, start + len(text), start + len(text), past_the_text=1)
        automation = FakeAutomation(FakeElement(FakeTextPattern([caret], field)))

        fitted = fit_to_context("Walk down to the orchard. ", _context_from(automation))

        assert fitted == "walk down to the orchard. "

    def test_a_selection_ending_exactly_where_the_field_starts_is_outside_it(self):
        page = _CHROME_BEFORE + "Hello there. " + _CHROME_AFTER
        start = len(_CHROME_BEFORE)
        field = FakeTextRange(page, start, len(page) - len(_CHROME_AFTER))
        automation = _automation_within(field, start - 4, start)

        assert _context_from(automation) is SELECTION_OUTSIDE_FIELD

    def test_should_read_the_full_width_when_it_fits_inside_the_field(self):
        field = "a" * 200 + "b" * 200
        automation = _automation_in_a_page(field, caret=200)

        context = _context_from(automation)

        assert context.before == "a" * CONTEXT_CHARS
        assert context.after == "b" * CONTEXT_CHARS


def _automation_with_a_line_starting_at_the_caret(before: str, line: str) -> FakeAutomation:
    """``before``, then a line starting at the caret with no character in between."""
    document = before + line
    field = FakeTextRange(document, 0, len(document))
    caret = FakeTextRange(document, len(before), len(before), line_starts=frozenset({len(before)}))
    return FakeAutomation(FakeElement(FakeTextPattern([caret], field)))


def _automation_on_a_new_paragraph(text: str) -> FakeAutomation:
    """``text``, then the empty paragraph Enter made, the way Chromium reads
    ProseMirror, Lexical, Quill or CodeMirror: nothing between the two, and
    the new paragraph's own trailing break after the caret."""
    return _automation_with_a_line_starting_at_the_caret(text, "\n")


class RangeThatCannotTellLines(FakeTextRange):
    def Clone(self) -> "RangeThatCannotTellLines":
        return RangeThatCannotTellLines(self.document, self.start, self.end)

    def ExpandToEnclosingUnit(self, unit: int) -> None:
        raise OSError("this provider has no line unit")


class TestACaretOnANewLine:
    def test_a_box_that_cannot_tell_lines_is_still_read(self):
        text = "Pack the lantern, and "
        caret = RangeThatCannotTellLines(text, len(text), len(text))
        automation = FakeAutomation(FakeElement(FakeTextPattern([caret], _whole_of(text))))

        assert _context_from(automation) == CaretContext(text, "", "")

    @pytest.mark.parametrize(
        "text",
        ["Pack the lantern.", "Things to pack", "x", "Ça va — naïve 東京 café", "y" * (CONTEXT_CHARS - 1), "z" * 300],
        ids=["finished sentence", "unfinished line", "one character", "unicode", "one short of the width", "long"],
    )
    def test_the_text_before_a_caret_on_a_new_paragraph_ends_with_a_line_break(self, text):
        context = _context_from(_automation_on_a_new_paragraph(text))

        assert context.before == (text + "\n")[-CONTEXT_CHARS:]
        assert context.after == "\n"

    @pytest.mark.parametrize("text", ["Pack the lantern.", "Things to pack", "Ça va, naïve"])
    def test_a_dictation_on_a_new_paragraph_starts_with_a_capital_and_nothing_in_front(self, text):
        fitted = fit_to_context("walk down. ", _context_from(_automation_on_a_new_paragraph(text)))

        assert fitted == "Walk down. "

    @pytest.mark.parametrize("text", ["Pack the lantern, and ", "Things to pack", "naïve 東京"])
    def test_a_caret_at_the_end_of_a_line_reads_no_line_break(self, text):
        context = _context_from(_automation_over(text, len(text), len(text)))

        assert context.before == text

    def test_a_caret_at_the_start_of_a_soft_wrapped_line_reads_no_line_break(self):
        automation = _automation_with_a_line_starting_at_the_caret(
            "Pack the lantern and ", "walk down to the orchard.",
        )

        assert _context_from(automation).before == "Pack the lantern and "

    def test_a_caret_at_the_end_whose_line_reads_as_nothing_reads_no_line_break(self):
        automation = _automation_with_a_line_starting_at_the_caret("Pack the lantern, and ", "")

        assert _context_from(automation).before == "Pack the lantern, and "

    @pytest.mark.parametrize("line_break", ["\n", "\r", "\r\n"])
    def test_a_line_break_already_read_before_the_caret_is_not_doubled(self, line_break):
        automation = _automation_with_a_line_starting_at_the_caret("Pack the lantern." + line_break, "\n")

        assert _context_from(automation).before == "Pack the lantern." + line_break


class TestWebViewsThatReportOnlyTheirHost:
    """WhatsApp's composer often reads as the blank pane around its WebView2."""

    def test_should_read_the_focused_field_inside_a_blank_web_view(self):
        field = _whole_of("Testing 1, 2, 3.")
        composer = FakeElement(FakeTextPattern([FakeTextRange(field.document, 16, 16)], field))
        bridge = FakeElement(
            NullPointer(), control_type=_PANE,
            class_name="Microsoft.UI.Xaml.Controls.WebView2", framework="Win32",
            focused_inside=composer,
        )

        context = _context_from(FakeAutomation(bridge))

        assert context.before == "Testing 1, 2, 3."
        assert context.after == ""

    def test_should_report_no_context_when_nothing_inside_the_web_view_has_the_focus(self):
        bridge = FakeElement(NullPointer(), control_type=_PANE, framework="Chrome")

        assert _context_from(FakeAutomation(bridge)) == NO_CONTEXT

    def test_should_not_search_inside_a_pane_that_is_not_a_web_view(self):
        game = FakeElement(
            NullPointer(), control_type=_PANE, class_name="UnityWndClass", framework="Win32",
            focused_inside=FakeElement(FakeTextPattern([], _whole_of("chat"))),
        )

        assert _context_from(FakeAutomation(game)) == NO_CONTEXT


class TestContextWhenNoneCanBeRead:
    def test_a_password_field_is_never_read(self):
        field = _whole_of(_DOCUMENT)
        selection = FakeTextRange(field.document, 0, len(_DOCUMENT))
        automation = FakeAutomation(
            FakeElement(FakeTextPattern([selection], field), is_password=True),
        )

        assert _context_from(automation) == NO_CONTEXT

    def test_should_report_no_context_when_the_element_has_no_text_pattern(self):
        automation = FakeAutomation(FakeElement(NullPointer()))

        assert _context_from(automation) == NO_CONTEXT

    def test_should_report_no_context_when_nothing_is_selected(self):
        automation = FakeAutomation(
            FakeElement(FakeTextPattern([], _whole_of(_DOCUMENT))),
        )

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
    def test_a_selection_outside_the_field_survives_the_edit_control_fallback(
        self, monkeypatch,
    ):
        page = _CHROME_BEFORE + "Hello there. " + _CHROME_AFTER
        field = FakeTextRange(page, len(_CHROME_BEFORE), len(page) - len(_CHROME_AFTER))
        automation = _automation_within(field, 0, 0)
        monkeypatch.setattr(module, "_automation", lambda: automation)
        monkeypatch.setattr(module, "read_edit_control_context", lambda: NO_CONTEXT)

        assert read_caret_context() is SELECTION_OUTSIDE_FIELD

    def test_should_read_the_focused_element(self, monkeypatch):
        automation = _automation_over(_DOCUMENT, _SELECTION_START, _SELECTION_END)
        monkeypatch.setattr(module, "_automation", lambda: automation)

        assert read_caret_context().selected == "here for water"

    def test_should_fall_back_to_the_edit_control_reader_without_a_text_pattern(
        self, monkeypatch,
    ):
        """A WinForms ``TextBox`` publishes no UI Automation text pattern."""
        monkeypatch.setattr(
            module, "_automation", lambda: FakeAutomation(FakeElement(NullPointer())),
        )
        monkeypatch.setattr(
            module,
            "read_edit_control_context",
            lambda: CaretContext("Hello, I'm ", "here for water", ". "),
        )

        assert read_caret_context().selected == "here for water"

    def test_should_keep_the_ui_automation_reading_when_it_finds_one(
        self, monkeypatch,
    ):
        automation = _automation_over(_DOCUMENT, _SELECTION_START, _SELECTION_END)
        monkeypatch.setattr(module, "_automation", lambda: automation)
        monkeypatch.setattr(
            module,
            "read_edit_control_context",
            lambda: CaretContext("wrong", "wrong", "wrong"),
        )

        assert read_caret_context().selected == "here for water"


@pytest.fixture()
def capture_config():
    """Caret logging wired to a config the way bootstrap wires it."""
    config = FakeConfig({"diagnostics.enabled": True, "diagnostics.capture_content": False})
    log_caret_text_when(functools.partial(content_recording_enabled, config))
    yield config
    log_caret_text_when(lambda: False)


class TestWhatReachesTheLog:
    _PRIVATE = "Dear Zoë, my PIN is 4471 — see you at the café. "

    def _logged_line(self, app_caplog) -> str:
        app_caplog.set_level(logging.DEBUG, logger="skryb")
        app_caplog.clear()
        _context_from(_automation_over(self._PRIVATE, 20, 20))
        return next(
            record.getMessage() for record in app_caplog.records
            if record.getMessage().startswith("Caret context")
        )

    def test_the_text_around_the_cursor_stays_out_of_the_log_while_content_capture_is_off(
        self, app_caplog, capture_config,
    ):
        line = self._logged_line(app_caplog)

        assert line == "Caret context: 20 characters before, 0 selected, 28 after"
        assert "4471" not in line and "Zoë" not in line

    def test_the_text_is_quoted_once_content_capture_is_turned_on(
        self, app_caplog, capture_config,
    ):
        capture_config.set("diagnostics.capture_content", True)

        line = self._logged_line(app_caplog)

        assert "'Dear Zoë, my PIN is '" in line
        assert "4471" in line
