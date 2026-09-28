"""Tests for the Win32 edit-control caret context reader.

The Win32 surface is hand-faked rather than mocked: a fake ``user32`` holds
one window's class name, text and selection and answers the three messages
from them, so the slicing these tests exercise is the real one. Nothing here
touches ``ctypes.windll`` except the one real-window test, which is Windows only.
"""

import sys

import pytest

from voicetotext.output import win32_edit_control_context as module
from voicetotext.output.caret_context import (
    CONTEXT_CHARS,
    NO_CONTEXT,
    CaretContext,
)
from voicetotext.output.win32_edit_control_context import (
    edit_control_context_of,
    read_edit_control_context,
)


_EM_GETSEL = 0x00B0
_WM_GETTEXT = 0x000D
_WM_GETTEXTLENGTH = 0x000E

_HWND = 0x1234
_FOREGROUND_HWND = 0x9999
_THREAD = 77

_EDIT_CLASS = "WindowsForms10.EDIT.app.0.2498891_r8_ad1"
_TEXT = "Hello, I'm here for water. "
_SELECTION_START = 11
_SELECTION_END = 25


class FakeUser32:
    """One focused edit window, answering only the messages it should."""

    def __init__(
        self,
        class_name: str = _EDIT_CLASS,
        text: str = _TEXT,
        start: int = _SELECTION_START,
        end: int = _SELECTION_END,
    ) -> None:
        self.class_name = class_name
        self.text = text
        self.start = start
        self.end = end
        self.messages: list[int] = []
        self.answers_messages = True
        self.reports_focus = True
        self.style = 0
        self.caret_hwnd: int | None = None
        self.answers_thread_info = True

    def GetForegroundWindow(self) -> int:  # noqa: N802 (Win32 name)
        return _FOREGROUND_HWND

    def GetWindowThreadProcessId(self, hwnd: int, pid: object) -> int:  # noqa: N802
        assert hwnd == _FOREGROUND_HWND
        return _THREAD

    def GetGUIThreadInfo(self, thread: int, info: object) -> int:  # noqa: N802
        assert thread == _THREAD
        info._obj.hwndFocus = _HWND if self.reports_focus else None
        info._obj.hwndCaret = self.caret_hwnd
        return 1 if self.answers_thread_info else 0

    def GetClassNameW(self, hwnd: int, buffer: object, length: int) -> int:  # noqa: N802
        buffer.value = self.class_name
        return len(self.class_name)

    def GetWindowLongW(self, hwnd: int, index: int) -> int:  # noqa: N802 (Win32 name)
        return self.style

    def SendMessageTimeoutW(  # noqa: N802 (Win32 name)
        self,
        hwnd: int,
        message: int,
        wparam: object,
        lparam: object,
        flags: int,
        timeout: int,
        result: object,
    ) -> int:
        self.messages.append(message)
        if not self.answers_messages:
            return 0
        if message == _EM_GETSEL:
            wparam._obj.value = self.start
            lparam._obj.value = self.end
        elif message == _WM_GETTEXTLENGTH:
            result._obj.value = len(self.text)
        elif message == _WM_GETTEXT:
            lparam.value = self.text
            result._obj.value = len(self.text)
        return 1


@pytest.fixture()
def user32(monkeypatch: pytest.MonkeyPatch) -> FakeUser32:
    fake = FakeUser32()
    monkeypatch.setattr(module, "_user32", lambda: fake)
    return fake


class TestContextFromAnEditControl:
    def test_should_split_the_text_around_the_selection(self, user32: FakeUser32):
        context = edit_control_context_of(_HWND)

        assert context.before == "Hello, I'm "
        assert context.selected == "here for water"
        assert context.after == ". "

    def test_should_report_an_empty_selection_for_a_bare_caret(
        self, user32: FakeUser32,
    ):
        user32.end = user32.start

        context = edit_control_context_of(_HWND)

        assert context.selected == ""
        assert context.before == "Hello, I'm "
        assert context.after == "here for water. "

    def test_should_read_no_more_than_the_context_width_on_each_side(
        self, user32: FakeUser32,
    ):
        user32.text = "a" * 200 + "SELECTED" + "b" * 200
        user32.start = 200
        user32.end = 208

        context = edit_control_context_of(_HWND)

        assert context.before == "a" * CONTEXT_CHARS
        assert context.after == "b" * CONTEXT_CHARS

    def test_should_keep_the_earlier_line_of_a_multi_line_control(
        self, user32: FakeUser32,
    ):
        user32.text = "line one\r\n" + _TEXT
        user32.start = _SELECTION_START + 10
        user32.end = _SELECTION_END + 10

        context = edit_control_context_of(_HWND)

        assert context.before == "line one\r\nHello, I'm "
        assert context.selected == "here for water"

    @pytest.mark.parametrize(
        "class_name", ["RICHEDIT50W", "WindowsForms10.RichEdit20W.app.0.1"],
    )
    def test_should_count_a_rich_edit_paragraph_break_as_one_character(
        self, user32: FakeUser32, class_name: str,
    ):
        user32.class_name = class_name
        user32.text = "one\r\ntwo\r\nthree"
        user32.start = user32.end = 8

        context = edit_control_context_of(_HWND)

        assert context.before == "one\rtwo\r"
        assert context.after == "three"

    @pytest.mark.parametrize(
        ("start", "end", "expected"),
        [
            (0, 0, CaretContext("", "", _TEXT)),
            (27, 27, CaretContext(_TEXT, "", "")),
            (25, 27, CaretContext("Hello, I'm here for water", ". ", "")),
            (0, 27, CaretContext("", _TEXT, "")),
        ],
        ids=["caret at the start", "caret at the end", "selection to the end", "everything"],
    )
    def test_should_read_up_to_the_edges_of_the_text(
        self, user32: FakeUser32, start: int, end: int, expected: CaretContext,
    ):
        user32.start = start
        user32.end = end

        assert edit_control_context_of(_HWND) == expected


class TestContextWhenNoneCanBeRead:
    def test_a_password_edit_control_is_never_asked_for_its_text(
        self, user32: FakeUser32,
    ):
        user32.style = 0x0020

        assert edit_control_context_of(_HWND) == NO_CONTEXT
        assert user32.messages == []

    def test_should_not_ask_a_window_that_is_not_an_edit_control_for_its_text(
        self, user32: FakeUser32,
    ):
        user32.class_name = "Chrome_WidgetWin_1"

        assert edit_control_context_of(_HWND) == NO_CONTEXT
        assert user32.messages == []

    def test_should_report_no_context_when_the_window_does_not_answer(
        self, user32: FakeUser32,
    ):
        user32.answers_messages = False

        assert edit_control_context_of(_HWND) == NO_CONTEXT

    def test_should_report_no_context_for_a_missing_window_handle(
        self, user32: FakeUser32,
    ):
        assert edit_control_context_of(0) == NO_CONTEXT
        assert user32.messages == []

    def test_should_report_no_context_when_the_win32_calls_fail(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        def explode() -> FakeUser32:
            raise OSError("user32.dll is missing")

        monkeypatch.setattr(module, "_user32", explode)

        assert read_edit_control_context() == NO_CONTEXT


class TestReadEditControlContext:
    def test_should_read_the_control_that_holds_the_keyboard_focus(
        self, user32: FakeUser32,
    ):
        assert read_edit_control_context().selected == "here for water"

    def test_should_report_no_context_when_no_control_has_the_focus(
        self, user32: FakeUser32,
    ):
        user32.reports_focus = False

        assert read_edit_control_context() == NO_CONTEXT
        assert user32.messages == []


class TestForegroundCaret:
    def test_a_foreground_app_showing_a_caret_has_a_text_field(
        self, user32: FakeUser32,
    ):
        user32.caret_hwnd = _HWND

        assert module.foreground_has_caret() is True

    def test_a_foreground_app_without_a_caret_has_none(self, user32: FakeUser32):
        assert module.foreground_has_caret() is False

    def test_an_unreadable_foreground_thread_is_an_error_not_a_guess(
        self, user32: FakeUser32,
    ):
        user32.answers_thread_info = False

        with pytest.raises(OSError):
            module.foreground_has_caret()
        assert read_edit_control_context() == NO_CONTEXT


@pytest.mark.skipif(sys.platform != "win32", reason="real Win32 windows")
def test_only_windows_directly_inside_count_and_never_the_whole_desktop(tk_root) -> None:
    import ctypes
    import tkinter

    from voicetotext.output.win32_edit_control_context import has_child_window_of_class

    window = tkinter.Toplevel(tk_root)
    try:
        tkinter.Frame(window).pack()
        button = tkinter.Button(window, text="after")
        button.pack()
        window.update()
        button.winfo_id()  # Tk makes a widget's native window on demand
        inner = window.winfo_id()
        wrapper = int(window.wm_frame(), 16)
        desktop = ctypes.windll.user32.GetDesktopWindow()

        assert has_child_window_of_class(inner, frozenset({"Button"}))
        assert has_child_window_of_class(inner, frozenset({"Chrome_WidgetWin_0", "Button"}))
        assert not has_child_window_of_class(inner, frozenset({"Chrome_WidgetWin_0"}))
        assert not has_child_window_of_class(wrapper, frozenset({"Button"}))
        assert not has_child_window_of_class(desktop, frozenset({"TkTopLevel"}))
    finally:
        window.destroy()
