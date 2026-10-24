"""Deciding whether the focused control can take a paste.

UI Automation and the Win32 caret are faked; nothing here imports
``comtypes`` or touches the desktop, so the file runs on any platform.
"""

from types import SimpleNamespace

import pytest

from skryb.output import win32_caret_context as module

_EDIT, _DOCUMENT, _BUTTON, _MENU_ITEM, _LIST_ITEM, _PANE = (
    50004, 50030, 50000, 50011, 50007, 50033,
)
_TEXT_PATTERN, _VALUE_PATTERN = 10014, 10002


class NullPointer:
    def __bool__(self) -> bool:
        return False


class FakeElement:
    def __init__(
        self,
        control_type: int,
        patterns: frozenset[int] = frozenset(),
        *,
        class_name: str = "",
        framework: str = "Win32",
        hwnd: int | None = None,
        focused_inside: "FakeElement | None" = None,
    ) -> None:
        self.CurrentControlType = control_type
        self.CurrentClassName = class_name
        self.CurrentFrameworkId = framework
        self.CurrentNativeWindowHandle = hwnd
        self._patterns = patterns
        self._focused_inside = focused_inside

    def GetCurrentPattern(self, pattern_id: int) -> object:  # noqa: N802 (COM name)
        return object() if pattern_id in self._patterns else NullPointer()

    def FindFirst(self, scope: int, condition: object) -> "FakeElement | None":  # noqa: N802 (COM name)
        return self._focused_inside


@pytest.fixture(autouse=True)
def stub_uia(monkeypatch) -> None:
    monkeypatch.setattr(module, "_uia", lambda: SimpleNamespace(
        UIA_EditControlTypeId=_EDIT,
        UIA_DocumentControlTypeId=_DOCUMENT,
        UIA_PaneControlTypeId=_PANE,
        UIA_HasKeyboardFocusPropertyId=30008,
        TreeScope_Descendants=4,
        UIA_TextPatternId=_TEXT_PATTERN,
        UIA_ValuePatternId=_VALUE_PATTERN,
    ))


_WHATSAPP_BRIDGE_HWND = 197492
_PLAIN_WINUI_BRIDGE_HWND = 5001
# The window classes directly inside each window, as FindWindowEx finds them.
_CHILD_WINDOWS = {
    _WHATSAPP_BRIDGE_HWND: {"Chrome_WidgetWin_0", "InputSiteWindowClass"},
    _PLAIN_WINUI_BRIDGE_HWND: {"InputSiteWindowClass"},
}


def _focus(monkeypatch, element: FakeElement | None, *, caret: bool = False) -> list[str]:
    asked: list[str] = []
    monkeypatch.setattr(
        module,
        "has_child_window_of_class",
        lambda hwnd, classes: bool(_CHILD_WINDOWS.get(hwnd, set()) & classes),
    )

    def automation() -> SimpleNamespace:
        asked.append("uia")
        if element is None:
            raise OSError("UIAutomationCore.dll is missing")
        return SimpleNamespace(
            GetFocusedElement=lambda: element,
            CreatePropertyCondition=lambda property_id, value: (property_id, value),
        )

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


@pytest.mark.parametrize(
    "element",
    [
        FakeElement(_PANE, class_name="EmbeddedBrowserTabRootView", framework="Chrome"),
        FakeElement(_PANE, class_name="BrowserRootView", framework="Chrome"),
        FakeElement(_PANE, class_name="Chrome_WidgetWin_1"),
        FakeElement(_PANE, class_name="Microsoft.UI.Xaml.Controls.WebView2"),
        FakeElement(_PANE, class_name="Chrome_RenderWidgetHostHWND"),
        FakeElement(
            _PANE, class_name="Microsoft.UI.Content.DesktopChildSiteBridge",
            hwnd=_WHATSAPP_BRIDGE_HWND,
        ),
    ],
    ids=[
        "whatsapp-tab-root", "whatsapp-browser-root", "electron-window", "winui-webview2",
        "chrome-render-widget", "whatsapp-winui-bridge-holding-the-web-view",
    ],
)
def test_a_web_view_whose_page_is_not_described_yet_is_still_pasted_into(
    monkeypatch, element,
) -> None:
    _focus(monkeypatch, element)

    assert module.text_field_focused() is True


@pytest.mark.parametrize(
    "element",
    [
        FakeElement(_BUTTON, framework="Chrome"),
        FakeElement(_LIST_ITEM, framework="Chrome"),
        FakeElement(_PANE, class_name="UnityWndClass", hwnd=9001),
        FakeElement(
            _PANE, class_name="Microsoft.UI.Content.DesktopChildSiteBridge",
            hwnd=_PLAIN_WINUI_BRIDGE_HWND,
        ),
    ],
    ids=["web-button", "web-chat-list-item", "game-window", "winui-host-without-web-view"],
)
def test_real_non_text_controls_in_or_near_web_views_are_still_not_pasted_into(
    monkeypatch, element,
) -> None:
    _focus(monkeypatch, element)

    assert module.text_field_focused() is False


def test_the_focused_element_inside_a_blank_web_view_decides(monkeypatch) -> None:
    def bridge(inside: FakeElement) -> FakeElement:
        return FakeElement(
            _PANE, class_name="Microsoft.UI.Content.DesktopChildSiteBridge",
            hwnd=_WHATSAPP_BRIDGE_HWND, focused_inside=inside,
        )

    _focus(monkeypatch, bridge(FakeElement(_EDIT, framework="Chrome")))
    assert module.text_field_focused() is True

    _focus(monkeypatch, bridge(FakeElement(_BUTTON, framework="Chrome")))
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
