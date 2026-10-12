"""Read the text around the caret in the focused window via UI Automation.

Read-only throughout: no window is activated and no input is injected, so
asking for the context can never disturb what the user is typing into.
"""

import logging
from functools import cache
from types import ModuleType
from typing import Protocol

from voicetotext.output.caret_context import (
    CONTEXT_CHARS,
    NO_CONTEXT,
    SELECTION_OUTSIDE_FIELD,
    CaretContext,
    caret_log_line,
)
from voicetotext.output.win32_edit_control_context import (
    has_child_window_of_class,
    foreground_has_caret,
    read_edit_control_context,
)


_logger = logging.getLogger(__name__)

# IUIAutomationTextRange.GetText takes a maximum length, where -1 means
# "the whole range".
_ALL_TEXT = -1

# CUIAutomation8, the class that also serves pre-Windows-8 clients.
_AUTOMATION_CLSID = "{E22AD333-B25F-460C-83D0-0581107395C9}"

# Chromium, Electron and WebView2 build a page's accessibility tree lazily,
# so until it exists the focus reads as a blank pane of the browser around
# the page, or of the window hosting it (WhatsApp's composer reads as its
# WinUI content bridge). That means "cannot tell", not "no text field".
_CHROMIUM_FRAMEWORK = "Chrome"
_WEB_VIEW_HOST_CLASSES = frozenset({
    "Chrome_WidgetWin_0",
    "Chrome_WidgetWin_1",
    "Chrome_RenderWidgetHostHWND",
    "Microsoft.UI.Xaml.Controls.WebView2",
})


class _TextRange(Protocol):
    def Clone(self) -> "_TextRange": ...  # noqa: N802 (COM name)
    def CompareEndpoints(  # noqa: N802 (COM name)
        self, endpoint: int, target: "_TextRange", target_endpoint: int,
    ) -> int: ...
    def GetText(self, max_length: int) -> str: ...  # noqa: N802 (COM name)
    def MoveEndpointByRange(  # noqa: N802 (COM name)
        self, endpoint: int, target: "_TextRange", target_endpoint: int,
    ) -> None: ...
    def MoveEndpointByUnit(  # noqa: N802 (COM name)
        self, endpoint: int, unit: int, count: int,
    ) -> int: ...


class _TextRangeArray(Protocol):
    @property
    def Length(self) -> int: ...  # noqa: N802 (COM name)
    def GetElement(self, index: int) -> _TextRange: ...  # noqa: N802 (COM name)


class _TextPattern(Protocol):
    @property
    def DocumentRange(self) -> _TextRange: ...  # noqa: N802 (COM name)
    def GetSelection(self) -> _TextRangeArray: ...  # noqa: N802 (COM name)


class _PatternHandle(Protocol):
    """What ``GetCurrentPattern`` hands back: an IUnknown to narrow."""

    def QueryInterface(self, interface: object) -> _TextPattern: ...  # noqa: N802 (COM name)


class _Element(Protocol):
    @property
    def CurrentIsPassword(self) -> int: ...  # noqa: N802 (COM name)
    @property
    def CurrentControlType(self) -> int: ...  # noqa: N802 (COM name)
    @property
    def CurrentClassName(self) -> str: ...  # noqa: N802 (COM name)
    @property
    def CurrentFrameworkId(self) -> str: ...  # noqa: N802 (COM name)
    @property
    def CurrentNativeWindowHandle(self) -> int | None: ...  # noqa: N802 (COM name)
    def GetCurrentPattern(  # noqa: N802 (COM name)
        self, pattern_id: int,
    ) -> _PatternHandle: ...
    def FindFirst(self, scope: int, condition: object) -> "_Element | None": ...  # noqa: N802 (COM name)


class _Automation(Protocol):
    def GetFocusedElement(self) -> _Element: ...  # noqa: N802 (COM name)
    def ElementFromHandle(self, hwnd: int) -> _Element: ...  # noqa: N802 (COM name)
    def CreatePropertyCondition(self, property_id: int, value: object) -> object: ...  # noqa: N802 (COM name)


def read_caret_context() -> CaretContext:
    """Text around the caret in the focused control, or ``NO_CONTEXT``.

    Never raises. UI Automation answers first because it reaches every modern
    framework. A WinForms ``TextBox`` publishes no text pattern, so the
    edit-control reader picks up what UI Automation cannot see.
    """
    context = read_ui_automation_context()
    if context != NO_CONTEXT:
        return context
    fallback = read_edit_control_context()
    # Kept as it is, so the caller can tell why nothing was read.
    return context if fallback == NO_CONTEXT else fallback


def read_ui_automation_context() -> CaretContext:
    """The UI Automation half of the reading on its own, or ``NO_CONTEXT``.

    Never raises. A window with no text pattern, a control that reports no
    selection, a COM failure and a missing UIAutomationCore.dll all read as
    "no context", which callers treat as ordinary data.
    """
    try:
        return _context_from(_automation())
    except Exception:
        _logger.debug("Reading the caret context failed.", exc_info=True)
        return NO_CONTEXT


def text_field_focused() -> bool:
    """Whether the keyboard focus is somewhere text can be pasted. Never raises.

    A system caret answers first and cheaply. Otherwise UI Automation is
    asked what the focused element is. When neither can answer, the paste
    goes ahead as it always did: an unreadable app must not lose dictation.
    """
    try:
        if foreground_has_caret():
            return True
    except OSError:
        _logger.debug("The foreground thread has no GUI state to read.")
    try:
        return focused_element_takes_text()
    except Exception:
        _logger.debug("UI Automation could not describe the focus.", exc_info=True)
        return True


def focused_element_takes_text() -> bool:
    """Whether the focused element is an edit field or document, or holds a value.

    A blank web view pane with no focused element found inside counts as
    one too: the page inside is not described yet. Raises when UI
    Automation cannot answer.
    """
    uia = _uia()
    element = _focused_element(_automation())
    if element.CurrentControlType in (
        uia.UIA_EditControlTypeId, uia.UIA_DocumentControlTypeId,
    ):
        return True
    if any(
        element.GetCurrentPattern(pattern)
        for pattern in (uia.UIA_TextPatternId, uia.UIA_ValuePatternId)
    ):
        return True
    if _is_blank_web_view(element):
        _logger.debug("The focus is a web view not described yet (%s).", _described(element))
        return True
    _logger.debug("The focus takes no text (%s).", _described(element))
    return False


def _focused_element(automation: "_Automation") -> "_Element":
    """The focused element, looking inside a web view that reports only itself.

    WhatsApp's composer often reads as the WinUI bridge around its WebView2,
    while the page inside still marks the composer as the one with the focus.
    """
    element = automation.GetFocusedElement()
    if not _is_blank_web_view(element):
        return element
    uia = _uia()
    inner = element.FindFirst(
        uia.TreeScope_Descendants,
        automation.CreatePropertyCondition(uia.UIA_HasKeyboardFocusPropertyId, True),
    )
    if not inner:
        return element
    _logger.debug(
        "The focus reads as %s; using the focused element inside it (%s).",
        _described(element), _described(inner),
    )
    return inner


def _described(element: "_Element") -> str:
    return (
        f"control type {element.CurrentControlType}, class {element.CurrentClassName!r},"
        f" framework {element.CurrentFrameworkId!r}"
    )


def _is_blank_web_view(element: "_Element") -> bool:
    return element.CurrentControlType == _uia().UIA_PaneControlTypeId and _is_web_view(element)


def _is_web_view(element: "_Element") -> bool:
    if (
        element.CurrentFrameworkId == _CHROMIUM_FRAMEWORK
        or element.CurrentClassName in _WEB_VIEW_HOST_CLASSES
    ):
        return True
    hwnd = element.CurrentNativeWindowHandle
    return bool(hwnd) and has_child_window_of_class(int(hwnd), _WEB_VIEW_HOST_CLASSES)


def caret_context_of_hwnd(hwnd: int) -> CaretContext:
    """Same reading, aimed at one window handle instead of the focused one.

    Lets the field probe exercise the range walk against a control that was
    never given the keyboard focus.
    """
    try:
        return _context_of(_automation().ElementFromHandle(hwnd))
    except Exception:
        _logger.debug("Reading the caret context of %s failed.", hwnd, exc_info=True)
        return NO_CONTEXT


def _context_from(automation: _Automation) -> CaretContext:
    return _context_of(_focused_element(automation))


def _context_of(element: _Element) -> CaretContext:
    if element.CurrentIsPassword:
        _logger.debug("No caret context: the focused element is a password field.")
        return NO_CONTEXT
    uia = _uia()
    handle = element.GetCurrentPattern(uia.UIA_TextPatternId)
    if not handle:
        _logger.debug("No caret context: the focused element has no text pattern.")
        return NO_CONTEXT
    pattern = handle.QueryInterface(uia.IUIAutomationTextPattern)
    ranges = pattern.GetSelection()
    if ranges.Length == 0:
        _logger.debug("No caret context: the text pattern reports no selection.")
        return NO_CONTEXT
    selection = ranges.GetElement(0)
    document = pattern.DocumentRange
    if not _lies_within(selection, document):
        _logger.debug("No caret context: the selection lies outside the field.")
        return SELECTION_OUTSIDE_FIELD
    context = CaretContext(
        before=_adjacent_text(
            selection, document, uia.TextPatternRangeEndpoint_Start,
        ),
        selected=selection.GetText(_ALL_TEXT),
        after=_adjacent_text(
            selection, document, uia.TextPatternRangeEndpoint_End,
        ),
    )
    _logger.debug("Caret context: %s", caret_log_line(context))
    return context


def _lies_within(selection: _TextRange, document: _TextRange) -> bool:
    """Whether the selection is the control's own, not the page's elsewhere.

    Chromium answers ``GetSelection`` on any text pattern with the page-wide
    selection, so an unfocused box reports a caret that sits outside it.
    """
    uia = _uia()
    start = uia.TextPatternRangeEndpoint_Start
    end = uia.TextPatternRangeEndpoint_End
    return (
        selection.CompareEndpoints(start, document, start) >= 0
        and selection.CompareEndpoints(end, document, end) <= 0
    )


def _adjacent_text(selection: _TextRange, document: _TextRange, edge: int) -> str:
    """The text just outside ``edge`` of the selection, ``CONTEXT_CHARS`` wide.

    Reads backwards from the start of the selection and forwards from its
    end, stopping at the edge of ``document``.
    """
    uia = _uia()
    reads_backwards = edge == uia.TextPatternRangeEndpoint_Start
    opposite_edge = (
        uia.TextPatternRangeEndpoint_End
        if reads_backwards
        else uia.TextPatternRangeEndpoint_Start
    )
    adjacent = selection.Clone()
    adjacent.MoveEndpointByRange(opposite_edge, selection, edge)
    adjacent.MoveEndpointByUnit(
        edge,
        uia.TextUnit_Character,
        -CONTEXT_CHARS if reads_backwards else CONTEXT_CHARS,
    )
    _clamp_to_document(adjacent, edge, document)
    return adjacent.GetText(_ALL_TEXT)


def _clamp_to_document(adjacent: _TextRange, edge: int, document: _TextRange) -> None:
    """Hold ``edge`` inside the text pattern's own document range.

    Chromium exposes one flat text tree per page, so a character walk out of
    an editable box keeps going, and the buttons and labels around a composer
    read as ordinary text. The pattern's document range is bounded to the
    control, and is the only boundary the provider offers.
    """
    uia = _uia()
    start = uia.TextPatternRangeEndpoint_Start
    end = uia.TextPatternRangeEndpoint_End
    if adjacent.CompareEndpoints(edge, document, start) < 0:
        adjacent.MoveEndpointByRange(edge, document, start)
    elif adjacent.CompareEndpoints(edge, document, end) > 0:
        adjacent.MoveEndpointByRange(edge, document, end)


def _automation() -> _Automation:
    """A fresh UI Automation client on the calling thread.

    Built per call because each caret read runs on its own short-lived
    thread, and a COM object belongs to the apartment that created it.
    """
    import comtypes
    import comtypes.client

    comtypes.CoInitializeEx()
    return comtypes.client.CreateObject(
        _AUTOMATION_CLSID, interface=_uia().IUIAutomation,
    )


@cache
def _uia() -> ModuleType:
    """The generated ``UIAutomationClient`` module.

    Imported lazily: ``GetModule`` writes the generated wrapper under
    ``comtypes/gen`` on first use and needs UIAutomationCore.dll, neither of
    which should be able to break importing this module.
    """
    import comtypes.client

    comtypes.client.GetModule("UIAutomationCore.dll")
    from comtypes.gen import UIAutomationClient

    return UIAutomationClient
