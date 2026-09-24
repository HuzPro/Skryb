import ctypes
import sys
import time
from unittest.mock import MagicMock

import pytest

from tests.fakes.dispatcher import InlineActionDispatcher
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    RecordingPausedEvent,
    RecordingResumedEvent,
    StateChangedEvent,
    TranscriptionStartedEvent,
)
from voicetotext.recording import overlay as overlay_module
from voicetotext.recording.overlay import (
    RecordingOverlay,
    Win32LayeredOverlayWindow,
    premultiplied_bgra,
)
from voicetotext.recording.pill import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    Button,
    Mode,
    button_centres,
    target_size,
)

_FRAME_SECONDS = 1 / 60


class FakeWindowApi:
    def __init__(self) -> None:
        self.shown_at: list[tuple[int, int]] = []
        self.frames: list[bytes] = []
        self.hidden = 0
        self.disposed = 0
        self.mouse = None
        self.cursor: tuple[int, int] | None = None
        self.hand_cursor = False

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

    def set_mouse_handler(self, handler) -> None:
        self.mouse = handler

    def cursor_position(self) -> tuple[int, int] | None:
        return self.cursor

    def set_hand_cursor(self, hand: bool) -> None:
        self.hand_cursor = hand


class FakeRoot:
    """Runs the overlay's frame timer on demand, against a fake clock."""

    def __init__(self) -> None:
        self.now = 100.0
        self.pending = None
        self.cancelled: list[str] = []

    def after(self, _ms: int, callback) -> str:
        self.pending = callback
        return "frame"

    def after_cancel(self, after_id: str) -> None:
        self.cancelled.append(after_id)
        self.pending = None

    def run(self, seconds: float) -> None:
        end = self.now + seconds
        while self.pending is not None and self.now < end:
            self.now += _FRAME_SECONDS
            callback, self.pending = self.pending, None
            callback()


class Harness:
    def __init__(self, estimate_level: float = 0.25) -> None:
        self.root = FakeRoot()
        self.bus = SimpleEventBus()
        self.api = FakeWindowApi()
        self.actions: list[str] = []
        self.presses = 0
        self.ticks = 0
        self.overlay = RecordingOverlay(
            root=self.root,
            event_bus=self.bus,
            dispatcher=InlineActionDispatcher(),
            level_provider=lambda: estimate_level,
            window_api=self.api,
            on_cancel=lambda: self.actions.append("cancel"),
            on_pause_toggle=lambda: self.actions.append("pause"),
            on_finish=lambda: self.actions.append("finish"),
            on_press=self._pressed,
            on_tick=self._ticked,
            clock=lambda: self.root.now,
        )

    def _pressed(self) -> None:
        self.presses += 1

    def _ticked(self) -> None:
        self.ticks += 1

    def state(self, old: str, new: str) -> None:
        self.bus.publish(StateChangedEvent(old_state=old, new_state=new))

    def start_recording(self) -> None:
        self.state("idle", "recording")
        self.root.run(1.0)

    def click(self, button: Button, release_on: Button | None = None) -> None:
        x, y, _ = button_centres(self.overlay.frame)[button]
        self.api.mouse(True, round(x), round(y))
        rx, ry, _ = button_centres(self.overlay.frame)[release_on or button]
        self.api.mouse(False, round(rx), round(ry))


class TestAppearing:
    def test_sits_at_the_bottom_centre_of_the_foreground_monitor(self):
        harness = Harness()

        harness.state("idle", "recording")

        x, y = harness.api.shown_at[0]
        assert x == 100 + (1000 - CANVAS_WIDTH) // 2
        assert y + CANVAS_HEIGHT < 850

    def test_grows_in_from_a_sliver_with_a_small_overshoot_then_settles(self):
        harness = Harness()
        full_width, full_height = target_size(Mode.RECORDING)
        widths = []

        harness.state("idle", "recording")
        for _ in range(60):
            harness.root.run(_FRAME_SECONDS)
            widths.append(harness.overlay.frame.width)

        assert widths[0] < full_width / 2
        assert full_width < max(widths) < full_width * 1.08
        assert widths[-1] == pytest.approx(full_width, abs=0.5)
        assert harness.overlay.frame.height == pytest.approx(full_height, abs=0.5)
        assert harness.overlay.frame.opacity == pytest.approx(1.0, abs=0.01)

    def test_paints_every_frame_at_the_full_canvas_size(self):
        harness = Harness()

        harness.start_recording()

        assert harness.api.frames
        assert {len(frame) for frame in harness.api.frames} == {
            CANVAS_WIDTH * CANVAS_HEIGHT * 4,
        }

    def test_checks_for_stray_clicks_on_every_frame(self):
        harness = Harness()

        harness.start_recording()

        assert harness.ticks == len(harness.api.frames)


class TestButtons:
    @pytest.mark.parametrize(
        ("button", "action"),
        [(Button.CANCEL, "cancel"), (Button.PAUSE, "pause"), (Button.FINISH, "finish")],
    )
    def test_each_button_does_its_job(self, button, action):
        harness = Harness()
        harness.start_recording()

        harness.click(button)

        assert harness.actions == [action]

    def test_a_press_dragged_off_to_another_button_does_nothing(self):
        harness = Harness()
        harness.start_recording()

        harness.click(Button.CANCEL, release_on=Button.FINISH)

        assert harness.actions == []

    def test_clicking_the_waveform_does_nothing_but_still_reports_the_press(self):
        harness = Harness()
        harness.start_recording()

        harness.api.mouse(True, CANVAS_WIDTH // 2 - 20, 38)
        harness.api.mouse(False, CANVAS_WIDTH // 2 - 20, 38)

        assert harness.actions == []
        assert harness.presses == 1

    def test_buttons_are_not_clickable_while_the_pill_is_still_growing(self):
        harness = Harness()
        harness.state("idle", "recording")

        harness.click(Button.FINISH)

        assert harness.actions == []

    def test_buttons_are_gone_once_transcribing(self):
        harness = Harness()
        harness.start_recording()
        finish = button_centres(harness.overlay.frame)[Button.FINISH]
        harness.state("recording", "transcribing")
        harness.root.run(1.0)

        harness.api.mouse(True, round(finish[0]), round(finish[1]))
        harness.api.mouse(False, round(finish[0]), round(finish[1]))

        assert harness.actions == []

    def test_hovering_a_button_shows_the_hand_cursor(self):
        harness = Harness()
        harness.start_recording()
        x, y, _ = button_centres(harness.overlay.frame)[Button.PAUSE]

        harness.api.cursor = (round(x), round(y))
        harness.root.run(0.1)

        assert harness.api.hand_cursor is True
        assert harness.overlay.frame.hovered is Button.PAUSE

    def test_the_cursor_goes_back_to_normal_off_the_buttons(self):
        harness = Harness()
        harness.start_recording()
        harness.api.cursor = (0, 0)

        harness.root.run(0.1)

        assert harness.api.hand_cursor is False


class TestPause:
    def test_a_paused_pill_flattens_its_waveform(self):
        harness = Harness(estimate_level=0.9)
        harness.start_recording()

        harness.bus.publish(RecordingPausedEvent())
        harness.root.run(0.2)

        assert harness.overlay.frame.paused is True
        assert harness.overlay.frame.level == 0.0

    def test_resuming_brings_the_waveform_back(self):
        harness = Harness(estimate_level=0.9)
        harness.start_recording()
        harness.bus.publish(RecordingPausedEvent())

        harness.bus.publish(RecordingResumedEvent())
        harness.root.run(0.2)

        assert harness.overlay.frame.paused is False
        assert harness.overlay.frame.level > 0.5

    def test_a_new_recording_never_starts_paused(self):
        harness = Harness()
        harness.start_recording()
        harness.bus.publish(RecordingPausedEvent())
        harness.state("recording", "idle")
        harness.root.run(1.0)

        harness.start_recording()

        assert harness.overlay.frame.paused is False


class TestTranscribingProgress:
    def _transcribing(self, estimate: float | None) -> Harness:
        harness = Harness()
        harness.start_recording()
        harness.state("recording", "transcribing")
        harness.bus.publish(
            TranscriptionStartedEvent(audio_seconds=4.0, estimated_seconds=estimate),
        )
        return harness

    def test_shrinks_into_the_transcribing_bar(self):
        harness = self._transcribing(2.0)

        harness.root.run(1.0)

        frame = harness.overlay.frame
        assert frame.width == pytest.approx(target_size(Mode.TRANSCRIBING)[0], abs=1)
        assert frame.controls == pytest.approx(0.0, abs=0.02)
        assert frame.label == pytest.approx(1.0, abs=0.02)

    def test_fills_in_step_with_the_learnt_estimate(self):
        harness = self._transcribing(2.0)

        harness.root.run(1.0)

        assert harness.overlay.frame.progress == pytest.approx(0.475, abs=0.04)

    def test_never_looks_finished_while_the_engine_is_still_running(self):
        harness = self._transcribing(0.5)

        harness.root.run(20.0)

        assert 0.95 <= harness.overlay.frame.progress < 1.0

    def test_still_moves_without_an_estimate_but_stops_short(self):
        harness = self._transcribing(None)

        harness.root.run(0.5)
        early = harness.overlay.frame.progress
        harness.root.run(30.0)

        assert 0.2 < early
        assert harness.overlay.frame.progress < 0.9

    def test_never_moves_backwards(self):
        harness = self._transcribing(3.0)
        seen = []

        for _ in range(200):
            harness.root.run(_FRAME_SECONDS)
            seen.append(harness.overlay.frame.progress)

        assert seen == sorted(seen)

    def test_fills_holds_then_leaves_when_the_text_is_in(self):
        harness = self._transcribing(5.0)
        harness.root.run(0.5)

        harness.state("transcribing", "idle")
        harness.root.run(0.1)
        assert harness.overlay.frame.progress > 0.8
        harness.root.run(2.0)

        assert harness.overlay.visible is False
        assert harness.api.hidden == 1
        assert harness.root.pending is None

    def test_leaves_the_same_way_after_a_failed_transcription(self):
        harness = self._transcribing(1.0)

        harness.state("transcribing", "error")
        harness.root.run(2.0)

        assert harness.overlay.visible is False


class TestLeaving:
    def test_cancelling_shrinks_the_pill_away_without_a_progress_bar(self):
        harness = Harness()
        harness.start_recording()

        harness.state("recording", "idle")
        harness.root.run(0.05)
        assert harness.overlay.frame.label == 0.0
        harness.root.run(2.0)

        assert harness.overlay.visible is False
        assert harness.api.hidden == 1

    def test_a_new_recording_mid_exit_comes_straight_back_in_place(self):
        harness = Harness()
        harness.start_recording()
        harness.state("recording", "idle")
        harness.root.run(0.05)

        harness.start_recording()

        assert harness.overlay.mode is Mode.RECORDING
        assert len(harness.api.shown_at) == 1
        assert harness.overlay.frame.width == pytest.approx(
            target_size(Mode.RECORDING)[0], abs=0.5,
        )

    def test_dispose_stops_listening_and_destroys_the_window(self):
        harness = Harness()
        harness.start_recording()

        harness.overlay.dispose()
        harness.state("idle", "recording")

        assert harness.api.disposed == 1
        assert harness.root.pending is None
        assert len(harness.api.shown_at) == 1


class TestMarshalling:
    def test_worker_thread_events_are_handed_to_the_gui_thread(self):
        dispatcher = MagicMock()
        bus = SimpleEventBus()
        RecordingOverlay(
            root=FakeRoot(), event_bus=bus, dispatcher=dispatcher,
            level_provider=lambda: 0.0, window_api=FakeWindowApi(),
        )

        bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))
        bus.publish(RecordingPausedEvent())
        bus.publish(TranscriptionStartedEvent(audio_seconds=1.0, estimated_seconds=0.3))

        assert dispatcher.dispatch.call_count == 3


class TestFramePixels:
    def test_premultiplied_buffer_is_bgra_with_no_channel_above_alpha(self):
        harness = Harness()
        harness.start_recording()

        pixels = harness.api.frames[-1]

        assert pixels[:4] == b"\x00\x00\x00\x00"
        assert all(
            pixels[index] <= pixels[index + 3]
            for index in range(0, len(pixels), 4)
        )

    def test_premultiplying_keeps_the_size(self):
        from PIL import Image

        image = Image.new("RGBA", (3, 2), (200, 100, 50, 128))

        assert len(premultiplied_bgra(image)) == 3 * 2 * 4


class TestWin32Window:
    def _window(self) -> Win32LayeredOverlayWindow:
        window = Win32LayeredOverlayWindow.__new__(Win32LayeredOverlayWindow)
        window._user32 = MagicMock()
        window._kernel32 = MagicMock()
        window._mouse_handler = None
        window._hand_cursor = False
        window._origin = (0, 0)
        return window

    def test_is_layered_topmost_and_never_activates_but_does_take_clicks(self):
        window = self._window()
        window._user32.CreateWindowExW.return_value = 4242

        assert window._create_window() == 4242

        ex_styles = window._user32.CreateWindowExW.call_args.args[0]
        assert ex_styles & 0x00080000
        assert ex_styles & 0x08000000
        assert ex_styles & 0x00000080
        assert ex_styles & 0x00000008
        assert not ex_styles & 0x00000020

    def test_refuses_to_be_activated_by_a_click(self):
        window = self._window()

        assert window.handle_message(0x0021, 0) == 3

    def test_reports_clicks_with_signed_coordinates(self):
        window = self._window()
        clicks = []
        window.set_mouse_handler(lambda *args: clicks.append(args))
        left_of_primary = (0xFFFF & -5) | (40 << 16)

        window.handle_message(0x0201, left_of_primary)
        window.handle_message(0x0202, (12 << 16) | 30)

        assert clicks == [(True, -5, 40), (False, 30, 12)]

    def test_leaves_every_other_message_to_windows(self):
        window = self._window()

        assert window.handle_message(0x0200, 0) is None

    def test_a_failing_handler_never_escapes_into_windows(self, monkeypatch):
        window = self._window()
        window.set_mouse_handler(MagicMock(side_effect=RuntimeError("boom")))
        monkeypatch.setitem(overlay_module._WINDOWS, 99, window)
        default = MagicMock(return_value=0)
        monkeypatch.setattr(overlay_module, "_default_window_proc", lambda: default)

        assert overlay_module._window_proc(99, 0x0201, 0, 0) == 0
        default.assert_called_once()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="Win32 desktop acceptance")
def test_clicking_finish_keeps_the_focus_in_the_app_being_dictated_into(
    tk_root,
) -> None:
    """Drives the real mouse: run it in stuntvm, never on a working desktop."""
    import tkinter

    target = tkinter.Toplevel(tk_root)
    target.geometry("400x200+100+100")
    target.deiconify()
    target.focus_force()
    target.update()
    user32 = ctypes.windll.user32
    target_hwnd = user32.GetParent(target.winfo_id()) or target.winfo_id()
    user32.SetForegroundWindow(target_hwnd)
    foreground_before = user32.GetForegroundWindow()
    finished: list[bool] = []
    api = Win32LayeredOverlayWindow()
    overlay = RecordingOverlay(
        root=tk_root,
        event_bus=SimpleEventBus(),
        dispatcher=InlineActionDispatcher(),
        level_provider=lambda: 0.2,
        window_api=api,
        on_finish=lambda: finished.append(True),
    )
    try:
        overlay.show()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            tk_root.update()
        x, y, _ = button_centres(overlay.frame)[Button.FINISH]
        user32.SetCursorPos(api._origin[0] + round(x), api._origin[1] + round(y))
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        deadline = time.monotonic() + 1.0
        while not finished and time.monotonic() < deadline:
            tk_root.update()
        assert finished == [True]
        assert user32.GetForegroundWindow() == foreground_before
    finally:
        overlay.dispose()
        target.destroy()
