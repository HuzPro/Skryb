"""Read the text around the caret in the focused window via UI Automation.

Read-only throughout: no window is activated and no input is injected, so
asking for the context can never disturb what the user is typing into.
"""

import logging
from functools import cache
from types import ModuleType
from typing import Protocol

from voicetotext.output.caret_context import NO_CONTEXT, CaretContext


_logger = logging.getLogger(__name__)

CONTEXT_CHARS = 40

# IUIAutomationTextRange.GetText takes a maximum length, where -1 means
# "the whole range".
_ALL_TEXT = -1

# CUIAutomation8, the class that also serves pre-Windows-8 clients.
_AUTOMATION_CLSID = "{E22AD333-B25F-460C-83D0-0581107395C9}"

_LOGGED_TEXT_CHARS = 20


class _TextRange(Protocol):
    def Clone(self) -> "_TextRange": ...  # noqa: N802 (COM name)
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
    def GetSelection(self) -> _TextRangeArray: ...  # noqa: N802 (COM name)


class _PatternHandle(Protocol):
    """What ``GetCurrentPattern`` hands back: an IUnknown to narrow."""

    def QueryInterface(self, interface: object) -> _TextPattern: ...  # noqa: N802 (COM name)


class _Element(Protocol):
    def GetCurrentPattern(  # noqa: N802 (COM name)
        self, pattern_id: int,
    ) -> _PatternHandle: ...


class _Automation(Protocol):
    def GetFocusedElement(self) -> _Element: ...  # noqa: N802 (COM name)
    def ElementFromHandle(self, hwnd: int) -> _Element: ...  # noqa: N802 (COM name)


def read_caret_context() -> CaretContext:
    """Text around the caret in the focused control, or ``NO_CONTEXT``.

    Never raises. A window with no text pattern, a control that reports no
    selection, a COM failure and a missing UIAutomationCore.dll all read as
    "no context", which callers treat as ordinary data.
    """
    try:
        return _context_from(_automation())
    except Exception:
        _logger.debug("Reading the caret context failed.", exc_info=True)
        return NO_CONTEXT


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
    return _context_of(automation.GetFocusedElement())


def _context_of(element: _Element) -> CaretContext:
    uia = _uia()
    handle = element.GetCurrentPattern(uia.UIA_TextPatternId)
    if not handle:
        return NO_CONTEXT
    pattern = handle.QueryInterface(uia.IUIAutomationTextPattern)
    ranges = pattern.GetSelection()
    if ranges.Length == 0:
        return NO_CONTEXT
    selection = ranges.GetElement(0)
    context = CaretContext(
        before=_adjacent_text(
            selection, uia.TextPatternRangeEndpoint_Start, -CONTEXT_CHARS,
        ),
        selected=selection.GetText(_ALL_TEXT),
        after=_adjacent_text(
            selection, uia.TextPatternRangeEndpoint_End, CONTEXT_CHARS,
        ),
    )
    _log_context(context)
    return context


def _adjacent_text(selection: _TextRange, edge: int, characters: int) -> str:
    """The text just outside ``edge`` of the selection, ``characters`` wide.

    A negative width reads backwards from the edge, a positive one forwards.
    """
    uia = _uia()
    opposite_edge = (
        uia.TextPatternRangeEndpoint_End
        if edge == uia.TextPatternRangeEndpoint_Start
        else uia.TextPatternRangeEndpoint_Start
    )
    adjacent = selection.Clone()
    adjacent.MoveEndpointByRange(opposite_edge, selection, edge)
    adjacent.MoveEndpointByUnit(edge, uia.TextUnit_Character, characters)
    return adjacent.GetText(_ALL_TEXT)


def _log_context(context: CaretContext) -> None:
    _logger.debug(
        "Caret context: before=%r selected=%r after=%r",
        context.before[-_LOGGED_TEXT_CHARS:],
        context.selected[:_LOGGED_TEXT_CHARS],
        context.after[:_LOGGED_TEXT_CHARS],
    )


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
