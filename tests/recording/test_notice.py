import sys
import time

import pytest

from voicetotext.core.color_scheme import ColorScheme
from voicetotext.recording import notice as notice_module
from voicetotext.recording.notice import Notice, Win32NoticeWindow

_WORK_AREA = (0, 0, 1920, 1040)


class FakeNoticeWindow:
    """Records what the notice asks of Windows; never shows anything."""

    def __init__(self) -> None:
        self.prepared: list[int] = []
        self.shown: list[tuple[int, int, int, int, int]] = []
        self.revealed: list[int] = []
        self.hidden: list[int] = []

    def prepare(self, hwnd: int) -> None:
        self.prepared.append(hwnd)

    def work_area(self) -> tuple[int, int, int, int]:
        return _WORK_AREA

    def show(self, hwnd: int, x: int, y: int, width: int, height: int) -> None:
        self.shown.append((hwnd, x, y, width, height))

    def reveal(self, hwnd: int) -> None:
        self.revealed.append(hwnd)

    def hide(self, hwnd: int) -> None:
        self.hidden.append(hwnd)


def _pump(tk_root, until, seconds: float = 3.0) -> None:
    deadline = time.monotonic() + seconds
    while not until() and time.monotonic() < deadline:
        tk_root.update()
        time.sleep(0.005)


@pytest.fixture()
def native() -> FakeNoticeWindow:
    return FakeNoticeWindow()


@pytest.fixture()
def notice(tk_root, native):
    toast = Notice(tk_root, native, color_scheme=lambda: ColorScheme.DARK)
    yield toast
    toast.dispose()


def _click(button) -> None:
    button.invoke()


def test_a_notice_is_built_hidden_and_shown_only_through_the_no_activate_path(
    notice, native,
) -> None:
    notice.show("Transcription failed.", [("Retry", lambda: None), ("Open history", lambda: None)])

    window = notice._window
    assert window.state() == "withdrawn"
    assert window.overrideredirect()
    assert notice.visible
    assert notice.message == "Transcription failed."
    assert [button.cget("text") for button in notice.buttons()] == ["Retry", "Open history"]
    assert all(str(button.cget("takefocus")) == "0" for button in notice.buttons())
    hwnd, x, y, width, height = native.shown[-1]
    assert native.prepared == [hwnd]
    assert x == (1920 - width) // 2
    assert y + height == 1040 - 26 - 72 - 8


def test_a_notice_stays_see_through_until_its_content_has_painted(
    tk_root, notice, native,
) -> None:
    notice.show("Recording cancelled.", [("Transcribe anyway", lambda: None)])
    assert native.revealed == []

    _pump(tk_root, lambda: native.revealed)

    assert native.revealed == [native.shown[-1][0]]


def test_a_notice_put_away_or_replaced_before_it_painted_is_revealed_only_as_the_latest(
    tk_root, notice, native,
) -> None:
    notice.show("Transcription failed.", [("Retry", lambda: None)])
    notice.hide()
    _pump(tk_root, lambda: False, seconds=0.1)
    assert native.revealed == []

    notice.show("Recording cancelled.", [("Transcribe anyway", lambda: None)])
    notice.show("Nothing to paste into. The text is on your clipboard.", [])
    _pump(tk_root, lambda: False, seconds=0.1)

    assert len(native.revealed) == 1


def test_buttons_run_their_action_and_put_the_notice_away(notice, native) -> None:
    pressed: list[str] = []
    notice.show("The paste did not land.", [
        ("Paste again", lambda: pressed.append("paste")),
        ("Copy", lambda: pressed.append("copy")),
    ])

    _click(notice.buttons()[1])

    assert pressed == ["copy"]
    assert not notice.visible
    assert len(native.hidden) == 1


def test_a_second_notice_replaces_the_first(notice, native) -> None:
    pressed: list[str] = []
    notice.show("Recording cancelled.", [("Transcribe anyway", lambda: pressed.append("old"))])
    first_buttons = notice.buttons()

    notice.show(
        "Nothing to paste into. The text is on your clipboard.",
        [("Open history", lambda: pressed.append("history"))],
    )

    assert notice.message.startswith("Nothing to paste into")
    assert [button.cget("text") for button in notice.buttons()] == ["Open history"]
    assert not any(button.winfo_exists() for button in first_buttons)
    assert len(native.shown) == 2
    _click(notice.buttons()[0])
    assert pressed == ["history"]


def test_at_most_three_buttons_are_offered(notice) -> None:
    notice.show("Choose", [(f"Option {n}", lambda: None) for n in range(5)])

    assert len(notice.buttons()) == 3


def test_a_notice_hides_itself_after_a_while(tk_root, native, monkeypatch) -> None:
    monkeypatch.setattr(notice_module, "_AUTO_HIDE_MS", 1)
    toast = Notice(tk_root, native, color_scheme=lambda: ColorScheme.LIGHT)
    try:
        toast.show("Recording cancelled.", [("Transcribe anyway", lambda: None)])
        deadline = time.monotonic() + 3.0
        while toast.visible and time.monotonic() < deadline:
            tk_root.update()
            time.sleep(0.005)

        assert not toast.visible
        assert len(native.hidden) == 1
    finally:
        toast.dispose()


def test_hiding_a_notice_that_is_not_showing_does_nothing(notice, native) -> None:
    notice.hide()
    notice.show("Transcription failed.", [])
    notice.hide()
    notice.hide()

    assert len(native.hidden) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 window styles")
def test_the_notice_window_is_marked_never_to_take_the_focus(tk_root, native) -> None:
    real = Win32NoticeWindow()

    class StylesOnly(FakeNoticeWindow):
        def prepare(self, hwnd: int) -> None:
            super().prepare(hwnd)
            real.prepare(hwnd)

    native = StylesOnly()
    toast = Notice(tk_root, native, color_scheme=lambda: ColorScheme.LIGHT)
    try:
        toast.show("Transcription failed.", [("Retry", lambda: None)])
        hwnd = native.prepared[-1]
        style = real._user32.GetWindowLongW(hwnd, notice_module._GWL_EXSTYLE)

        assert style & notice_module._WS_EX_NOACTIVATE
        assert style & notice_module._WS_EX_TOOLWINDOW
        assert real._user32.GetForegroundWindow() != hwnd
    finally:
        toast.dispose()
