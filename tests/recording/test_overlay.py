import ctypes
import itertools
import sys
import time
from unittest.mock import MagicMock

import pytest

from tests.fakes.dispatcher import InlineActionDispatcher
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    MicrophoneAudioResumedEvent,
    MicrophoneSilentEvent,
    RecordingCapWarningEvent,
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
    BOTTOM_MARGIN,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_PILL_POSITION,
    Button,
    Mode,
    button_centres,
    target_size,
)

_FRAME_SECONDS = 1 / 60
# A spot on the pill between its buttons.
_WAVEFORM = (CANVAS_WIDTH // 2 - 20, 38)


@pytest.fixture(autouse=True)
def restore_switch_interval():
    """A test that leaves a pill on screen must not speed up GIL hand-offs for the rest."""
    before = sys.getswitchinterval()
    yield
    sys.setswitchinterval(before)


class FakeWindowApi:
    def __init__(self) -> None:
        self.shown_at: list[tuple[int, int]] = []
        self.frames: list[bytes] = []
        self.hidden = 0
        self.disposed = 0
        self.mouse = None
        self.cursor: tuple[int, int] | None = None
        self.hand_cursor = False
        self.frame_waits = 0
        self.synced = True
        self.work_area = (100, 50, 1100, 850)
        self.origin = (0, 0)
        self.moved_to: list[tuple[int, int]] = []
        self.button_down = False

    def work_area_for_foreground(self) -> tuple[int, int, int, int]:
        return self.work_area

    def show_at(self, x: int, y: int) -> None:
        self.shown_at.append((x, y))
        self.origin = (x, y)

    def move_to(self, x: int, y: int) -> None:
        # The cursor stays put on screen, so against the window it moves back.
        if self.cursor is not None:
            self.cursor = (
                self.cursor[0] - (x - self.origin[0]),
                self.cursor[1] - (y - self.origin[1]),
            )
        self.moved_to.append((x, y))
        self.origin = (x, y)

    def left_button_down(self) -> bool:
        return self.button_down

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

    def wait_for_frame(self) -> bool:
        self.frame_waits += 1
        return self.synced


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
    def __init__(
        self,
        estimate_level: float = 0.25,
        position: tuple[float, float] = DEFAULT_PILL_POSITION,
    ) -> None:
        self.root = FakeRoot()
        self.bus = SimpleEventBus()
        self.api = FakeWindowApi()
        self.actions: list[str] = []
        self.presses = 0
        self.ticks = 0
        self.position = position
        self.saved: list[tuple[float, float]] = []
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
            load_position=lambda: self.position,
            save_position=self.saved.append,
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

    def press(self, x: int = _WAVEFORM[0], y: int = _WAVEFORM[1]) -> None:
        self.api.button_down = True
        self.api.cursor = (x, y)
        self.api.mouse(True, x, y)

    def move_by(self, dx: int, dy: int) -> None:
        x, y = self.api.cursor
        self.api.cursor = (x + dx, y + dy)
        self.root.run(_FRAME_SECONDS * 2)

    def release(self) -> None:
        self.api.button_down = False
        self.api.mouse(False, *self.api.cursor)

    def drag_by(self, dx: int, dy: int) -> None:
        self.press()
        self.move_by(dx, dy)
        self.release()

    def origin(self) -> tuple[int, int]:
        return self.api.origin


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

    def test_checks_for_stray_clicks_at_most_twenty_times_a_second(self):
        harness = Harness()

        harness.start_recording()

        assert 14 <= harness.ticks <= 22
        assert harness.ticks < len(harness.api.frames)


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

        assert harness.overlay.frame.progress == pytest.approx(0.45, abs=0.03)

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
        assert harness.overlay.frame.progress <= 0.9

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


class TestPlacement:
    def test_first_shows_exactly_where_it_always_has(self):
        harness = Harness()

        harness.start_recording()

        assert harness.api.shown_at == [
            (100 + (1000 - CANVAS_WIDTH) // 2, 850 - CANVAS_HEIGHT - BOTTOM_MARGIN),
        ]

    def test_shows_at_the_remembered_place_on_the_foreground_monitor(self):
        harness = Harness(position=(0.0, 0.0))
        harness.api.work_area = (-1920, 0, 0, 1040)

        harness.start_recording()

        assert harness.api.shown_at == [(-1920, 0)]


class TestDragging:
    def test_dragging_the_pill_moves_it_and_remembers_where_it_was_left(self):
        harness = Harness()
        harness.start_recording()
        start = harness.origin()

        harness.drag_by(-200, -300)

        assert harness.origin() == (start[0] - 200, start[1] - 300)
        [(across, down)] = harness.saved
        assert across == pytest.approx((start[0] - 200 - 100) / (1000 - CANVAS_WIDTH))
        assert down == pytest.approx(
            (start[1] - 300 - 50) / (800 - CANVAS_HEIGHT - BOTTOM_MARGIN),
        )

    def test_the_pill_follows_the_cursor_all_the_way(self):
        harness = Harness()
        harness.start_recording()
        start = harness.origin()
        harness.press()

        for _ in range(30):
            harness.move_by(-7, -5)

        assert harness.origin() == (start[0] - 210, start[1] - 150)
        assert harness.saved == []
        harness.release()
        assert len(harness.saved) == 1

    def test_it_stops_at_the_edges_of_the_work_area(self):
        harness = Harness()
        harness.start_recording()

        harness.drag_by(-5000, 5000)

        assert harness.origin() == (100, 850 - CANVAS_HEIGHT - BOTTOM_MARGIN)
        assert harness.saved == [(0.0, 1.0)]

    def test_a_drag_never_presses_a_button(self):
        harness = Harness()
        harness.start_recording()
        finish = button_centres(harness.overlay.frame)[Button.FINISH]
        harness.press()

        harness.move_by(round(finish[0]) - _WAVEFORM[0], 0)
        harness.move_by(0, 20)
        harness.move_by(0, -20)
        harness.api.button_down = False
        harness.api.mouse(False, round(finish[0]), round(finish[1]))

        assert harness.actions == []

    def test_a_press_on_a_button_never_drags_the_pill(self):
        harness = Harness()
        harness.start_recording()
        pause = button_centres(harness.overlay.frame)[Button.PAUSE]
        harness.press(round(pause[0]), round(pause[1]))

        harness.move_by(-150, -150)
        harness.release()

        assert harness.api.moved_to == []
        assert harness.actions == []

    def test_a_wobble_while_clicking_is_not_a_drag(self):
        harness = Harness()
        harness.start_recording()

        harness.drag_by(2, -2)

        assert harness.api.moved_to == []
        assert harness.saved == []

    def test_letting_go_off_the_pill_still_ends_the_drag(self):
        harness = Harness()
        harness.start_recording()
        harness.press()
        harness.move_by(-100, 0)

        harness.api.button_down = False
        harness.move_by(-100, 0)

        assert len(harness.saved) == 1
        assert len(harness.api.moved_to) == 1

    def test_the_next_recording_comes_back_where_the_pill_was_left(self):
        harness = Harness()
        harness.start_recording()
        harness.drag_by(-300, -200)
        harness.position = harness.saved[-1]
        moved = harness.origin()
        harness.state("recording", "idle")
        harness.root.run(2.0)

        harness.start_recording()

        assert harness.api.shown_at[-1] == moved

    def test_dragging_works_while_transcribing_too(self):
        harness = Harness()
        harness.start_recording()
        harness.state("recording", "transcribing")
        harness.root.run(0.5)

        harness.drag_by(0, -100)

        assert len(harness.saved) == 1


class TestDoubleClickReset:
    def test_a_double_click_puts_the_pill_back_at_the_bottom_centre(self):
        harness = Harness(position=(0.1, 0.2))
        harness.start_recording()

        harness.drag_by(0, 0)
        harness.root.run(0.2)
        harness.drag_by(0, 0)

        assert harness.origin() == (100 + (1000 - CANVAS_WIDTH) // 2, 752)
        assert harness.saved == [DEFAULT_PILL_POSITION]
        assert harness.actions == []

    def test_two_slow_clicks_are_not_a_double_click(self):
        harness = Harness(position=(0.1, 0.2))
        harness.start_recording()

        harness.drag_by(0, 0)
        harness.root.run(0.8)
        harness.drag_by(0, 0)

        assert harness.saved == []

    def test_a_drag_then_a_click_is_not_a_double_click(self):
        harness = Harness()
        harness.start_recording()

        harness.drag_by(-100, 0)
        harness.drag_by(0, 0)

        assert len(harness.saved) == 1
        assert harness.saved[0] != DEFAULT_PILL_POSITION

    def test_double_clicking_a_button_presses_it_twice_and_leaves_the_pill_be(self):
        harness = Harness(position=(0.1, 0.2))
        harness.start_recording()

        harness.click(Button.PAUSE)
        harness.click(Button.PAUSE)

        assert harness.actions == ["pause", "pause"]
        assert harness.saved == []


class TestSilentMicrophone:
    def test_a_silent_microphone_replaces_the_waveform_with_its_name(self):
        harness = Harness()
        harness.start_recording()

        harness.bus.publish(MicrophoneSilentEvent(
            device_name="Microphone (Maono PD300X Microphone)", seconds_silent=4.0,
        ))
        harness.root.run(0.1)

        assert harness.overlay.frame.silent_microphone == "Maono PD300X Microphone"

    def test_the_waveform_comes_back_with_the_audio(self):
        harness = Harness()
        harness.start_recording()
        harness.bus.publish(MicrophoneSilentEvent(device_name="Windows default"))

        harness.bus.publish(MicrophoneAudioResumedEvent())
        harness.root.run(0.1)

        assert harness.overlay.frame.silent_microphone == ""

    def test_a_new_recording_starts_without_the_notice(self):
        harness = Harness()
        harness.start_recording()
        harness.bus.publish(MicrophoneSilentEvent(device_name="Windows default"))
        harness.state("recording", "idle")
        harness.root.run(2.0)

        harness.start_recording()

        assert harness.overlay.frame.silent_microphone == ""


class TestLimitCountdown:
    def _warned(self) -> Harness:
        harness = Harness()
        harness.start_recording()
        harness.bus.publish(RecordingCapWarningEvent(remaining_seconds=30.0))
        return harness

    def _after(self, harness: Harness, seconds: float) -> str:
        """Let ``seconds`` pass between two frames, as a busy machine might."""
        harness.root.now += seconds
        harness.root.run(_FRAME_SECONDS)
        return harness.overlay.frame.countdown

    def test_counts_down_the_time_left_before_the_limit(self):
        harness = self._warned()

        assert self._after(harness, 0.0) == "0:30"
        assert self._after(harness, 10.0) == "0:20"
        assert self._after(harness, 9.5) == "0:11"

    def test_the_countdown_stands_still_while_paused(self):
        harness = self._warned()
        self._after(harness, 5.0)

        harness.bus.publish(RecordingPausedEvent())
        paused = self._after(harness, 20.0)
        harness.bus.publish(RecordingResumedEvent())

        assert paused == "0:25"
        assert self._after(harness, 5.0) == "0:20"

    def test_never_counts_below_zero(self):
        harness = self._warned()

        assert self._after(harness, 45.0) == "0:00"

    def test_a_new_recording_has_no_countdown(self):
        harness = self._warned()
        harness.state("recording", "idle")
        harness.root.run(2.0)

        harness.start_recording()

        assert harness.overlay.frame.countdown == ""


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

    def test_moving_never_activates_resizes_or_restacks_the_pill(self):
        window = self._window()
        window._hwnd = 7

        window.move_to(-40, 300)

        flags = window._user32.SetWindowPos.call_args.args[-1]
        assert flags & 0x0010
        assert flags & 0x0001
        assert flags & 0x0004
        assert window._origin == (-40, 300)

    @pytest.mark.parametrize(("swapped", "button"), [(0, 0x01), (1, 0x02)])
    def test_reads_the_primary_button_even_when_the_buttons_are_swapped(
        self, swapped, button,
    ):
        window = self._window()
        window._user32.GetSystemMetrics.return_value = swapped
        window._user32.GetAsyncKeyState.return_value = -32768

        assert window.left_button_down() is True
        window._user32.GetAsyncKeyState.assert_called_once_with(button)

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


class TestSmoothness:
    def _transcribing(self, estimate: float | None = 2.0) -> Harness:
        harness = Harness()
        harness.start_recording()
        harness.state("recording", "transcribing")
        harness.bus.publish(
            TranscriptionStartedEvent(audio_seconds=4.0, estimated_seconds=estimate),
        )
        harness.root.run(1.0)
        return harness

    def test_every_frame_waits_for_the_display_refresh(self):
        harness = Harness()

        harness.start_recording()

        assert harness.api.frame_waits == len(harness.api.frames)

    def test_asks_to_come_back_quickly_when_synced_to_the_display(self, monkeypatch):
        harness = Harness()
        delays = []
        monkeypatch.setattr(
            harness.root, "after", lambda ms, callback: delays.append(ms) or "frame",
        )

        harness.state("idle", "recording")

        assert delays == [overlay_module._FRAME_POLL_MS]

    def test_falls_back_to_a_plain_timer_without_a_display_refresh(self, monkeypatch):
        harness = Harness()
        harness.api.synced = False
        delays = []
        monkeypatch.setattr(
            harness.root, "after", lambda ms, callback: delays.append(ms) or "frame",
        )

        harness.state("idle", "recording")

        assert delays == [overlay_module._UNSYNCED_FRAME_MS]

    def test_a_settled_bar_is_filled_without_redrawing_the_pill(self, monkeypatch):
        harness = self._transcribing()
        redraws = []
        original = overlay_module.render
        monkeypatch.setattr(
            overlay_module, "render", lambda frame: redraws.append(frame) or original(frame),
        )

        harness.root.run(0.5)

        assert redraws == []
        assert len(harness.api.frames) > 20

    def test_the_fill_advances_evenly_frame_to_frame(self):
        harness = self._transcribing(estimate=4.0)
        seen = []

        for _ in range(60):
            harness.root.run(_FRAME_SECONDS)
            seen.append(harness.overlay.frame.progress)

        steps = [after - before for before, after in itertools.pairwise(seen)]
        assert min(steps) > 0
        assert max(steps) / min(steps) < 1.2

    def test_the_finish_eases_to_full_then_the_pill_leaves(self):
        harness = self._transcribing(estimate=5.0)
        start = harness.overlay.frame.progress

        harness.state("transcribing", "idle")
        harness.root.run(0.1)
        midway = harness.overlay.frame.progress
        harness.root.run(0.15)
        full = harness.overlay.frame.progress
        harness.root.run(1.0)

        assert start < midway < full
        assert full == pytest.approx(1.0)
        assert harness.overlay.visible is False

    def test_restores_the_thread_switch_interval_once_the_pill_is_gone(self):
        before = sys.getswitchinterval()
        harness = Harness()

        harness.start_recording()
        during = sys.getswitchinterval()
        harness.state("recording", "idle")
        harness.root.run(2.0)

        assert during == pytest.approx(overlay_module._ANIMATING_SWITCH_INTERVAL)
        assert sys.getswitchinterval() == pytest.approx(before)

    def test_disposing_mid_animation_restores_the_switch_interval_too(self):
        before = sys.getswitchinterval()
        harness = Harness()
        harness.start_recording()

        harness.overlay.dispose()

        assert sys.getswitchinterval() == pytest.approx(before)


class TestWindowPixels:
    def test_copies_frames_straight_into_the_window_bitmap(self):
        import numpy

        window = Win32LayeredOverlayWindow.__new__(Win32LayeredOverlayWindow)
        target = numpy.zeros(CANVAS_WIDTH * CANVAS_HEIGHT * 4, dtype=numpy.uint8)
        window._pixels = target
        window._user32 = MagicMock()
        window._hwnd = 1
        window._screen_dc = window._memory_dc = 0
        frame = bytes(range(256)) * (len(target) // 256)

        window.push_frame(frame)

        assert target.tobytes() == frame
        window._user32.UpdateLayeredWindow.assert_called_once()

    def test_draws_nothing_before_it_has_a_bitmap(self):
        window = Win32LayeredOverlayWindow.__new__(Win32LayeredOverlayWindow)
        window._pixels = None
        window._user32 = MagicMock()

        window.push_frame(bytes(1) * (CANVAS_WIDTH * CANVAS_HEIGHT * 4))

        window._user32.UpdateLayeredWindow.assert_not_called()
