import ctypes
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from tests.fakes.dispatcher import InlineActionDispatcher
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import StateChangedEvent
from voicetotext.core.state_machine import AppState
from voicetotext.recording.overlay import (
    RecordingOverlay,
    Win32OverlayWindowApi,
)


class FakeWindowApi:
    def __init__(self) -> None:
        self.styled: list[int] = []

    def work_area_for_foreground(self) -> tuple[int, int, int, int]:
        return (100, 50, 1100, 850)

    def make_click_through(self, hwnd: int) -> None:
        self.styled.append(hwnd)


def _overlay() -> tuple[
    RecordingOverlay,
    SimpleEventBus,
    MagicMock,
    MagicMock,
    FakeWindowApi,
]:
    window = MagicMock()
    window.winfo_id.return_value = 99
    window.after.return_value = "timer-1"
    canvas = MagicMock()
    canvas.create_rectangle.side_effect = range(7)
    bus = SimpleEventBus()
    dispatcher = MagicMock()
    api = FakeWindowApi()
    with (
        patch("voicetotext.recording.overlay.tkinter.Toplevel", return_value=window),
        patch("voicetotext.recording.overlay.tkinter.Canvas", return_value=canvas),
    ):
        overlay = RecordingOverlay(
            root=MagicMock(),
            event_bus=bus,
            dispatcher=dispatcher,
            level_provider=lambda: 0.25,
            window_api=api,
        )
    return overlay, bus, dispatcher, window, api


class TestRecordingOverlayLifecycle:
    def test_should_marshal_show_and_hide_for_recording_transitions(self) -> None:
        overlay, bus, dispatcher, _window, _api = _overlay()

        bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))
        bus.publish(StateChangedEvent(old_state="recording", new_state="error"))

        actions = [item.args[0] for item in dispatcher.dispatch.call_args_list]
        assert actions[0].__self__ is overlay
        assert actions[0].__name__ == "show"
        assert actions[1].__self__ is overlay
        assert actions[1].__name__ == "hide"

    def test_should_position_style_and_start_timer_on_show(self) -> None:
        overlay, _bus, _dispatcher, window, api = _overlay()

        overlay.show()

        window.geometry.assert_called_once_with("112x48+544+766")
        assert api.styled == [99]
        window.deiconify.assert_called_once_with()
        window.after.assert_called_once()

    def test_should_cancel_timer_and_reset_levels_on_hide(self) -> None:
        overlay, _bus, _dispatcher, window, _api = _overlay()
        overlay.show()
        overlay._level = 0.8

        overlay.hide()

        window.after_cancel.assert_called_once_with("timer-1")
        window.withdraw.assert_called()
        assert overlay._level == 0.0

    def test_dispose_should_unsubscribe_and_destroy(self) -> None:
        overlay, bus, dispatcher, window, _api = _overlay()

        overlay.dispose()
        bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))

        dispatcher.dispatch.assert_not_called()
        window.destroy.assert_called_once_with()


class TestClickThroughStyles:
    def test_should_combine_layered_transparent_noactivate_and_toolwindow(self) -> None:
        user32 = MagicMock()
        user32.GetParent.return_value = 123
        user32.GetWindowLongW.return_value = 0x10
        user32.SetLayeredWindowAttributes.return_value = 1
        api = Win32OverlayWindowApi.__new__(Win32OverlayWindowApi)
        api._user32 = user32

        api.make_click_through(99)

        styles = user32.SetWindowLongW.call_args.args[2]
        assert styles & 0x00080000
        assert styles & 0x00000020
        assert styles & 0x08000000
        assert styles & 0x00000080


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="Win32 desktop acceptance")
def test_overlay_should_not_take_focus_and_click_should_reach_window_beneath(
    tk_root,
) -> None:
    import tkinter

    api = Win32OverlayWindowApi()
    left, _top, right, bottom = api.work_area_for_foreground()
    x = left + ((right - left - 112) // 2)
    y = bottom - 48 - 36
    target = tkinter.Toplevel(tk_root)
    # Undecorated, so its client area lands exactly under the overlay rather
    # than a title bar's height below it.
    target.overrideredirect(True)
    target.geometry(f"112x48+{x}+{y}")
    clicked: list[bool] = []
    target.bind("<Button-1>", lambda _event: clicked.append(True))
    target.deiconify()
    target.lift()
    target.focus_force()
    target.update()
    user32 = ctypes.windll.user32
    target_hwnd = user32.GetParent(target.winfo_id()) or target.winfo_id()
    user32.SetForegroundWindow(target_hwnd)
    foreground_before = user32.GetForegroundWindow()
    overlay = RecordingOverlay(
        root=tk_root,
        event_bus=SimpleEventBus(),
        dispatcher=InlineActionDispatcher(),
        level_provider=lambda: 0.2,
    )
    try:
        overlay.show()
        tk_root.update()
        assert user32.GetForegroundWindow() == foreground_before
        user32.SetCursorPos(x + 56, y + 24)
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        deadline = time.monotonic() + 1.0
        while not clicked and time.monotonic() < deadline:
            tk_root.update()
        assert clicked == [True]
    finally:
        overlay.dispose()
        target.destroy()
