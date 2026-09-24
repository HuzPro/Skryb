import math

import pytest

from voicetotext.recording.pill import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    Button,
    Frame,
    Mode,
    Spring,
    button_centres,
    hit_test,
    pill_box,
    render,
    target_size,
    transcription_progress,
)


def _recording_frame(**changes) -> Frame:
    width, height = target_size(Mode.RECORDING)
    fields = {"opacity": 1.0, "controls": 1.0, "label": 0.0, **changes}
    return Frame(width=width, height=height, **fields)


class TestSpring:
    def test_settles_on_its_target(self):
        spring = Spring(0.0)
        spring.target = 100.0

        for _ in range(120):
            spring.step(1 / 60)

        assert spring.value == pytest.approx(100.0, abs=0.01)
        assert spring.settled()

    def test_is_stable_even_with_a_long_stalled_frame(self):
        spring = Spring(0.0)
        spring.target = 1.0

        spring.step(0.5)

        assert math.isfinite(spring.value)
        assert abs(spring.value) < 2.0

    def test_with_heavy_damping_never_overshoots(self):
        spring = Spring(0.0, damping=40.0)
        spring.target = 1.0
        values = []

        for _ in range(120):
            spring.step(1 / 60)
            values.append(spring.value)

        assert max(values) <= 1.0 + 1e-6


class TestProgress:
    @pytest.mark.parametrize(
        ("elapsed", "estimate", "expected"),
        [(0.0, 2.0, 0.0), (1.0, 2.0, 0.475), (2.0, 2.0, 0.95)],
    )
    def test_is_linear_against_the_estimate_up_to_ninety_five_percent(
        self, elapsed, estimate, expected,
    ):
        assert transcription_progress(elapsed, estimate) == pytest.approx(expected)

    def test_creeps_but_never_finishes_when_the_estimate_is_blown(self):
        values = [transcription_progress(t, 0.5) for t in (0.6, 1.0, 5.0, 3600.0)]

        assert values == sorted(values)
        assert all(0.95 < value < 1.0 for value in values)

    @pytest.mark.parametrize("estimate", [None, 0.0, -1.0])
    def test_eases_short_of_ninety_percent_without_a_usable_estimate(self, estimate):
        assert transcription_progress(0.4, estimate) > 0.2
        assert transcription_progress(3600.0, estimate) <= 0.9

    def test_treats_a_clock_that_ran_backwards_as_just_started(self):
        assert transcription_progress(-3.0, 2.0) == 0.0


class TestLayout:
    def test_the_pill_grows_upward_from_a_fixed_bottom_edge(self):
        small = pill_box(*target_size(Mode.CLOSED))
        large = pill_box(*target_size(Mode.RECORDING))

        assert small[3] == large[3]
        assert large[1] < small[1]

    def test_every_size_fits_the_canvas(self):
        for mode in Mode:
            left, top, right, bottom = pill_box(*target_size(mode))
            assert left >= 0 and right <= CANVAS_WIDTH
            assert top >= 0 and bottom <= CANVAS_HEIGHT

    def test_buttons_sit_cancel_then_pause_then_finish_without_touching(self):
        centres = button_centres(_recording_frame())
        cancel = centres[Button.CANCEL]
        pause = centres[Button.PAUSE]
        finish = centres[Button.FINISH]

        assert cancel[0] < pause[0] < finish[0]
        assert finish[0] - pause[0] > pause[2] + finish[2]

    @pytest.mark.parametrize("button", list(Button))
    def test_each_button_answers_a_click_on_its_centre(self, button):
        frame = _recording_frame()
        x, y, _ = button_centres(frame)[button]

        assert hit_test(frame, x, y) is button

    def test_nothing_answers_a_click_on_the_transparent_margin(self):
        assert hit_test(_recording_frame(), 1, 1) is None

    def test_half_faded_controls_do_not_answer_clicks(self):
        frame = _recording_frame(controls=0.5)
        x, y, _ = button_centres(frame)[Button.FINISH]

        assert hit_test(frame, x, y) is None


class TestRender:
    def test_draws_nothing_when_fully_faded(self):
        image = render(Frame(width=200, height=40, opacity=0.0, controls=1.0, label=0.0))

        assert image.getextrema()[3] == (0, 0)

    def test_the_corners_stay_transparent_so_clicks_pass_through(self):
        image = render(_recording_frame(level=0.8, phase=2.0))

        assert image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1))[3] == 0

    def test_the_waveform_moves_with_the_voice(self):
        quiet = render(_recording_frame(level=0.0, phase=1.0))
        loud = render(_recording_frame(level=1.0, phase=1.0))

        assert quiet.tobytes() != loud.tobytes()

    def test_a_paused_pill_looks_different_from_a_recording_one(self):
        recording = render(_recording_frame(level=0.5, phase=1.0))
        paused = render(_recording_frame(level=0.5, phase=1.0, paused=True))

        assert recording.tobytes() != paused.tobytes()

    def test_the_progress_fill_grows_across_the_transcribing_bar(self):
        width, height = target_size(Mode.TRANSCRIBING)
        left, _top, right, bottom = pill_box(width, height)
        probe = (round(left + (right - left) * 0.3), round(bottom - 3))

        def pixel(progress: float) -> tuple[int, ...]:
            frame = Frame(width, height, 1.0, 0.0, 1.0, progress=progress)
            return render(frame).getpixel(probe)

        assert pixel(0.0) != pixel(0.6)
        assert pixel(0.6) == pixel(0.9)

    @pytest.mark.parametrize("hovered", list(Button))
    def test_hovering_a_button_changes_how_it_looks(self, hovered):
        assert (
            render(_recording_frame(hovered=hovered)).tobytes()
            != render(_recording_frame()).tobytes()
        )

    @pytest.mark.parametrize("controls", [0.0, 0.2, 0.5, 0.8])
    def test_fading_buttons_never_punch_holes_through_the_pill(self, controls):
        frame = Frame(width=180, height=40, opacity=1.0, controls=controls, label=0.5)
        for x, y, _ in button_centres(frame).values():
            assert render(frame).getpixel((round(x), round(y)))[3] > 240
