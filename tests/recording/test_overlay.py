import ctypes
import sys
import time
from unittest.mock import MagicMock

import pytest

from tests.fakes.dispatcher import InlineActionDispatcher
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import StateChangedEvent
from voicetotext.recording.overlay import (
    RecordingOverlay,
    Win32LayeredOverlayWindow,
    premultiplied_bgra,
    render_overlay_frame,
)


_WIDTH = 148
_HEIGHT = 44


class FakeWindowApi:
    def __init__(self) -> None:
        self.shown_at: list[tuple[int, int]] = []
        self.frames: list[bytes] = []
        self.hidden = 0
        self.disposed = 0

    def work_area_for_foreground(self) -> tuple[int, int, int, int]:
        return (100, 50, 1100, 850)

    def show_at(self, x: int, y: int) -> None:
        self.shown_at.append((x, y))

    def push_frame(self, pixels: bytes) -> None:
        self.frames.append(pixels)

    def hide(self) -> None:
        self.hidden += 1

    def dispose(self) -> None:
        self.disposed += 1


def _overlay() -> tuple[
    RecordingOverlay,
    SimpleEventBus,
    MagicMock,
    MagicMock,
    FakeWindowApi,
]:
    root = MagicMock()
    root.after.return_value = "timer-1"
    bus = SimpleEventBus()
    dispatcher = MagicMock()
    api = FakeWindowApi()
    overlay = RecordingOverlay(
        root=root,
        event_bus=bus,
        dispatcher=dispatcher,
        level_provider=lambda: 0.25,
        window_api=api,
    )
    return overlay, bus, dispatcher, root, api


class TestRecordingOverlayLifecycle:
    def test_should_marshal_show_and_hide_for_recording_transitions(self) -> None:
        overlay, bus, dispatcher, _root, _api = _overlay()

        bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))
        bus.publish(StateChangedEvent(old_state="recording", new_state="error"))

        actions = [item.args[0] for item in dispatcher.dispatch.call_args_list]
        assert actions[0].__self__ is overlay
        assert actions[0].__name__ == "show"
        assert actions[1].__self__ is overlay
        assert actions[1].__name__ == "hide"

    def test_should_position_paint_and_start_timer_on_show(self) -> None:
        overlay, _bus, _dispatcher, root, api = _overlay()

        overlay.show()

        assert api.shown_at == [(526, 770)]
        assert len(api.frames) == 1
        assert len(api.frames[0]) == _WIDTH * _HEIGHT * 4
        root.after.assert_called_once()

    def test_should_cancel_timer_and_reset_levels_on_hide(self) -> None:
        overlay, _bus, _dispatcher, root, api = _overlay()
        overlay.show()
        overlay._level = 0.8

        overlay.hide()

        root.after_cancel.assert_called_once_with("timer-1")
        assert api.hidden == 1
        assert overlay._level == 0.0

    def test_dispose_should_unsubscribe_and_destroy(self) -> None:
        overlay, bus, dispatcher, _root, api = _overlay()

        overlay.dispose()
        bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))

        dispatcher.dispatch.assert_not_called()
        assert api.disposed == 1


class TestOverlayFrameRendering:
    def test_should_render_transparent_corners_and_an_opaque_centre(self) -> None:
        frame = render_overlay_frame(level=0.6, phase=1.0)

        assert frame.size == (_WIDTH, _HEIGHT)
        assert frame.getpixel((0, 0))[3] == 0
        assert frame.getpixel((_WIDTH - 1, _HEIGHT - 1))[3] == 0
        assert frame.getpixel((_WIDTH // 2, _HEIGHT // 2))[3] > 0

    def test_premultiplied_buffer_should_be_bgra_with_no_channel_over_alpha(
        self,
    ) -> None:
        frame = render_overlay_frame(level=0.6, phase=1.0)

        pixels = premultiplied_bgra(frame)

        assert len(pixels) == _WIDTH * _HEIGHT * 4
        assert pixels[:4] == b"\x00\x00\x00\x00"
        assert all(
            pixels[index] <= pixels[index + 3]
            for index in range(0, len(pixels), 4)
        )

    def test_idle_level_should_still_animate_between_phases(self) -> None:
        first = premultiplied_bgra(render_overlay_frame(level=0.0, phase=0.0))
        later = premultiplied_bgra(render_overlay_frame(level=0.0, phase=9.0))

        assert first != later


class TestWin32WindowStyles:
    def test_should_combine_layered_transparent_noactivate_and_toolwindow(self) -> None:
        window = Win32LayeredOverlayWindow.__new__(Win32LayeredOverlayWindow)
        window._user32 = MagicMock()
        window._kernel32 = MagicMock()
        window._user32.CreateWindowExW.return_value = 4242

        assert window._create_window() == 4242

        ex_styles = window._user32.CreateWindowExW.call_args.args[0]
        assert ex_styles & 0x00080000
        assert ex_styles & 0x00000020
        assert ex_styles & 0x08000000
        assert ex_styles & 0x00000080
        assert ex_styles & 0x00000008


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="Win32 desktop acceptance")
def test_overlay_should_not_take_focus_and_click_should_reach_window_beneath(
    tk_root,
) -> None:
    import tkinter

    api = Win32LayeredOverlayWindow()
    left, _top, right, bottom = api.work_area_for_foreground()
    x = left + ((right - left - _WIDTH) // 2)
    y = bottom - _HEIGHT - 36
    target = tkinter.Toplevel(tk_root)
    # Undecorated, so its client area lands exactly under the overlay rather
    # than a title bar's height below it.
    target.overrideredirect(True)
    # Topmost, so whatever the desktop happens to have parked at the bottom of
    # the work area cannot sit between the overlay and this window.
    target.attributes("-topmost", True)
    target.geometry(f"{_WIDTH}x{_HEIGHT}+{x}+{y}")
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
        window_api=api,
    )
    try:
        overlay.show()
        tk_root.update()
        assert user32.GetForegroundWindow() == foreground_before
        user32.SetCursorPos(x + (_WIDTH // 2), y + (_HEIGHT // 2))
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        deadline = time.monotonic() + 1.0
        while not clicked and time.monotonic() < deadline:
            tk_root.update()
        assert clicked == [True]
    finally:
        overlay.dispose()
        target.destroy()
