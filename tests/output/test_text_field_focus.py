"""Deciding whether the focused control can take a paste.

UI Automation and the Win32 caret are faked; nothing here imports
``comtypes`` or touches the desktop, so the file runs on any platform.
"""

from types import SimpleNamespace

import pytest

from voicetotext.output import win32_caret_context as module

_EDIT, _DOCUMENT, _BUTTON, _MENU_ITEM, _LIST_ITEM, _PANE = (
    50004, 50030, 50000, 50011, 50007, 50033,
)
_TEXT_PATTERN, _VALUE_PATTERN = 10014, 10002


class NullPointer:
    def __bool__(self) -> bool:
        return False


class FakeElement:
    def __init__(self, control_type: int, patterns: frozenset[int] = frozenset()) -> None:
        self.CurrentControlType = control_type
        self._patterns = patterns

    def GetCurrentPattern(self, pattern_id: int) -> object:  # noqa: N802 (COM name)
        return object() if pattern_id in self._patterns else NullPointer()


@pytest.fixture(autouse=True)
def stub_uia(monkeypatch) -> None:
    monkeypatch.setattr(module, "_uia", lambda: SimpleNamespace(
        UIA_EditControlTypeId=_EDIT,
        UIA_DocumentControlTypeId=_DOCUMENT,
        UIA_TextPatternId=_TEXT_PATTERN,
        UIA_ValuePatternId=_VALUE_PATTERN,
    ))


def _focus(monkeypatch, element: FakeElement | None, *, caret: bool = False) -> list[str]:
    asked: list[str] = []

    def automation() -> SimpleNamespace:
        asked.append("uia")
        if element is None:
            raise OSError("UIAutomationCore.dll is missing")
        return SimpleNamespace(GetFocusedElement=lambda: element)

    monkeypatch.setattr(module, "foreground_has_caret", lambda: caret)
    monkeypatch.setattr(module, "_automation", automation)
    return asked


def test_a_blinking_caret_is_enough_and_ui_automation_is_not_asked(monkeypatch) -> None:
    asked = _focus(monkeypatch, FakeElement(_PANE), caret=True)

    assert module.text_field_focused() is True
    assert asked == []


@pytest.mark.parametrize(
    "element",
    [
        FakeElement(_EDIT),
        FakeElement(_DOCUMENT),
        FakeElement(_PANE, frozenset({_TEXT_PATTERN})),
        FakeElement(_LIST_ITEM, frozenset({_VALUE_PATTERN})),
    ],
    ids=["edit", "document", "pane-with-text", "combo-with-value"],
)
def test_fields_that_take_text_are_pasted_into(monkeypatch, element) -> None:
    _focus(monkeypatch, element)

    assert module.text_field_focused() is True


@pytest.mark.parametrize(
    "element",
    [FakeElement(_BUTTON), FakeElement(_MENU_ITEM), FakeElement(_LIST_ITEM), FakeElement(_PANE)],
    ids=["button", "menu-item", "desktop-icon", "game-surface"],
)
def test_targets_without_a_text_field_are_not_pasted_into(monkeypatch, element) -> None:
    _focus(monkeypatch, element)

    assert module.text_field_focused() is False


def test_when_ui_automation_cannot_answer_the_paste_goes_ahead_as_before(
    monkeypatch,
) -> None:
    _focus(monkeypatch, None)

    assert module.text_field_focused() is True


def test_an_unreadable_foreground_thread_falls_back_to_ui_automation(monkeypatch) -> None:
    _focus(monkeypatch, FakeElement(_MENU_ITEM))

    def no_gui_state() -> bool:
        raise OSError("GetGUIThreadInfo failed")

    monkeypatch.setattr(module, "foreground_has_caret", no_gui_state)

    assert module.text_field_focused() is False
