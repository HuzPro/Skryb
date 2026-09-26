import itertools
import math

import numpy
import pytest

from voicetotext.recording.pill import (
    BOTTOM_MARGIN,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_PILL_POSITION,
    Button,
    Frame,
    Mode,
    SettledBar,
    Spring,
    VoiceLevel,
    button_centres,
    countdown_text,
    finishing_progress,
    is_settled_bar,
    hit_test,
    pill_box,
    pill_origin,
    pill_position,
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
        [(0.0, 2.0, 0.0), (0.5, 2.0, 0.225), (1.0, 2.0, 0.45)],
    )
    def test_moves_at_a_steady_pace_set_by_the_estimate(
        self, elapsed, estimate, expected,
    ):
        assert transcription_progress(elapsed, estimate) == pytest.approx(expected)

    @pytest.mark.parametrize("estimate", [0.05, 0.4, 2.0, 30.0])
    def test_speed_never_jumps_anywhere_along_the_bar(self, estimate):
        step = estimate / 2000
        values = [transcription_progress(i * step, estimate) for i in range(8000)]
        speeds = [after - before for before, after in itertools.pairwise(values)]

        jumps = [abs(after - before) for before, after in itertools.pairwise(speeds)]
        assert max(jumps) < max(speeds) * 0.01

    def test_is_most_of_the_way_there_when_the_estimate_runs_out(self):
        assert 0.8 < transcription_progress(2.0, 2.0) < 0.9

    def test_creeps_but_never_finishes_when_the_estimate_is_blown(self):
        values = [transcription_progress(t, 0.5) for t in (0.6, 1.0, 5.0, 3600.0)]

        assert values == sorted(values)
        assert all(value < 1.0 for value in values)
        assert values[-1] == pytest.approx(0.99)


class TestFinishingProgress:
    def test_runs_from_where_the_bar_was_to_full(self):
        assert finishing_progress(0.6, 0.0, 0.2) == pytest.approx(0.6)
        assert finishing_progress(0.6, 0.2, 0.2) == 1.0
        assert finishing_progress(0.6, 5.0, 0.2) == 1.0

    def test_slows_into_the_end(self):
        early = finishing_progress(0.0, 0.05, 0.2) - finishing_progress(0.0, 0.0, 0.2)
        late = finishing_progress(0.0, 0.2, 0.2) - finishing_progress(0.0, 0.15, 0.2)

        assert early > late > 0

    def test_a_zero_length_finish_is_simply_full(self):
        assert finishing_progress(0.3, 0.0, 0.0) == 1.0


class TestSettledBar:
    def _bar(self):
        from voicetotext.recording.overlay import premultiplied_array

        return SettledBar(premultiplied_array), premultiplied_array

    @pytest.mark.parametrize("progress", [0.0, 1.0])
    def test_matches_the_full_drawing_when_empty_and_when_full(self, progress):
        bar, to_pixels = self._bar()
        width, height = target_size(Mode.TRANSCRIBING)
        drawn = to_pixels(
            render(Frame(width, height, 1.0, 0.0, 1.0, progress=progress)),
        ).tobytes()

        assert bar.pixels(progress) == drawn

    def test_the_edge_moves_by_fractions_of_a_pixel(self):
        bar, _ = self._bar()

        frames = {bar.pixels(0.5 + i * 0.0005) for i in range(8)}

        assert len(frames) == 8

    def test_ignores_progress_outside_the_bar(self):
        bar, _ = self._bar()

        assert bar.pixels(-3.0) == bar.pixels(0.0)
        assert bar.pixels(7.0) == bar.pixels(1.0)

    def test_is_only_used_once_the_pill_has_fully_settled(self):
        width, height = target_size(Mode.TRANSCRIBING)

        assert is_settled_bar(Frame(width, height, 1.0, 0.0, 1.0))
        assert not is_settled_bar(Frame(width + 2, height, 1.0, 0.0, 1.0))
        assert not is_settled_bar(Frame(width, height, 0.9, 0.0, 1.0))
        assert not is_settled_bar(Frame(width, height, 1.0, 0.2, 1.0))

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


class TestVoiceLevel:
    def _settle(self, level: VoiceLevel, rms: float, seconds: float = 10.0) -> float:
        value = 0.0
        for _ in range(round(seconds * 30)):
            value = level.update(rms, 1 / 30)
        return value

    def test_speech_from_across_the_room_fills_the_bars_like_speech_up_close(self):
        near = self._settle(VoiceLevel(), 0.08)
        far = self._settle(VoiceLevel(), 0.012)

        assert far == pytest.approx(near, abs=0.05)
        assert far > 0.9

    def test_room_noise_below_the_gate_draws_flat(self):
        assert self._settle(VoiceLevel(), 0.002) == 0.0

    def test_a_short_pause_between_words_does_not_rescale(self):
        level = VoiceLevel()
        self._settle(level, 0.05)
        self._settle(level, 0.0, seconds=0.5)

        assert level.update(0.05, 1 / 30) > 0.9

    def test_a_sudden_shout_never_overflows_the_bars(self):
        level = VoiceLevel()
        self._settle(level, 0.01)

        assert level.update(0.9, 1 / 30) == 1.0

    @pytest.mark.parametrize("rms", [math.nan, -1.0, math.inf])
    def test_bad_readings_never_break_it(self, rms):
        level = VoiceLevel()

        value = level.update(rms, 1 / 30)

        assert 0.0 <= value <= 1.0
        assert 0.0 <= level.update(0.05, 1 / 30) <= 1.0


def test_a_caption_is_drawn_above_the_controls_without_moving_them():
    plain = _recording_frame()
    captioned = _recording_frame(caption="Editing selection")

    assert render(plain).tobytes() != render(captioned).tobytes()
    assert button_centres(plain) == button_centres(captioned)
    assert hit_test(captioned, *button_centres(captioned)[Button.FINISH][:2]) is Button.FINISH


# Work areas as Windows reports them: (left, top, right, bottom), taskbar
# already taken off, some monitors left of or above the primary one.
_WORK_AREAS = [
    (0, 0, 1920, 1040),
    (0, 0, 1921, 1033),
    (-2560, -360, 0, 1080),
    (1920, 0, 3360, 860),
    (0, 0, 1366, 728),
    (3840, 0, 7680, 2120),
]


def _old_placement(work_area: tuple[int, int, int, int]) -> tuple[int, int]:
    left, _top, right, bottom = work_area
    return (
        left + ((right - left - CANVAS_WIDTH) // 2),
        bottom - CANVAS_HEIGHT - BOTTOM_MARGIN,
    )


class TestPillPosition:
    @pytest.mark.parametrize("work_area", _WORK_AREAS)
    def test_the_default_sits_exactly_where_the_pill_always_has(self, work_area):
        assert pill_origin(DEFAULT_PILL_POSITION, work_area) == _old_placement(work_area)

    @pytest.mark.parametrize("work_area", _WORK_AREAS)
    def test_every_pixel_on_a_monitor_survives_being_saved_and_restored(self, work_area):
        left, top, right, bottom = work_area
        for x in range(left, right - CANVAS_WIDTH + 1, 7):
            for y in range(top, bottom - CANVAS_HEIGHT - BOTTOM_MARGIN + 1, 53):
                assert pill_origin(pill_position((x, y), work_area), work_area) == (x, y)

    def test_a_spot_on_one_monitor_lands_on_the_same_spot_of_another(self):
        laptop = (0, 0, 1366, 728)
        big_screen = (-2560, -360, 0, 1080)
        across_laptop = pill_origin((0.25, 0.4), laptop)

        position = pill_position(across_laptop, laptop)
        x, y = pill_origin(position, big_screen)

        assert position == pytest.approx((0.25, 0.4), abs=0.002)
        assert (x - -2560) / (2560 - CANVAS_WIDTH) == pytest.approx(0.25, abs=0.001)
        assert (y - -360) / (1440 - CANVAS_HEIGHT - BOTTOM_MARGIN) == pytest.approx(
            0.4, abs=0.001,
        )

    @pytest.mark.parametrize(
        ("origin", "expected"),
        [
            ((-9999, -9999), (0.0, 0.0)),
            ((9999, 9999), (1.0, 1.0)),
            ((-9999, 500), (0.0, 500 / (1040 - CANVAS_HEIGHT - BOTTOM_MARGIN))),
        ],
    )
    def test_off_the_edge_means_at_the_edge(self, origin, expected):
        assert pill_position(origin, (0, 0, 1920, 1040)) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "position", [(-3.0, 0.5), (0.5, 7.5), (math.inf, -math.inf)],
    )
    def test_nonsense_positions_still_keep_the_pill_on_screen(self, position):
        left, top, right, bottom = 0, 0, 1920, 1040
        x, y = pill_origin(position, (left, top, right, bottom))

        assert left <= x <= right - CANVAS_WIDTH
        assert top <= y <= bottom - CANVAS_HEIGHT

    def test_a_screen_narrower_than_the_pill_pins_it_to_the_corner(self):
        assert pill_origin((0.7, 0.7), (10, 20, 100, 60)) == (10, 20)
        assert pill_position((10, 20), (10, 20, 100, 60)) == DEFAULT_PILL_POSITION


class TestCountdownText:
    @pytest.mark.parametrize(
        ("seconds", "text"),
        [
            (30.0, "0:30"),
            (29.2, "0:30"),
            (0.4, "0:01"),
            (0.0, "0:00"),
            (-5.0, "0:00"),
            (61.0, "1:01"),
            (600.0, "10:00"),
            (math.nan, "0:00"),
        ],
    )
    def test_rounds_up_to_whole_seconds_as_m_ss(self, seconds, text):
        assert countdown_text(seconds) == text


def _amber(frame: Frame) -> numpy.ndarray:
    r, g, b, a = numpy.asarray(render(frame)).astype(int).transpose(2, 0, 1)
    return (a > 200) & (r > 200) & (g > 120) & (g < 210) & (b < 110)


def _amber_pixels(frame: Frame) -> int:
    return int(_amber(frame).sum())


class TestNotices:
    def test_a_silent_microphone_is_named_in_amber_instead_of_the_waveform(self):
        plain = _recording_frame(level=1.0, phase=1.0)
        silent = _recording_frame(
            level=1.0, phase=1.0, silent_microphone="Maono PD300X Microphone",
        )

        assert _amber_pixels(plain) == 0
        assert _amber_pixels(silent) > 50

    @pytest.mark.parametrize(
        "name",
        ["", "Gerät für Café ✓", "USB " * 60, "\u200b\u200b", "x"],
    )
    def test_any_microphone_name_fits_between_the_buttons(self, name):
        frame = _recording_frame(silent_microphone=name or "Windows default")
        centres = button_centres(frame)
        cancel_x, _y, radius = centres[Button.CANCEL]
        pause_x = centres[Button.PAUSE][0]

        columns = numpy.flatnonzero(_amber(frame).any(axis=0))

        assert cancel_x + radius < columns.min()
        assert columns.max() < pause_x - radius

    def test_the_countdown_sits_beside_a_narrower_waveform(self):
        plain = render(_recording_frame(level=1.0, phase=1.0))
        counting = render(_recording_frame(level=1.0, phase=1.0, countdown="0:30"))

        assert _amber_pixels(_recording_frame(countdown="0:30")) > 20
        assert plain.tobytes() != counting.tobytes()

    def test_a_silent_microphone_outranks_the_countdown(self):
        both = _recording_frame(silent_microphone="Blue Yeti", countdown="0:30")
        notice = _recording_frame(silent_microphone="Blue Yeti")

        assert render(both).tobytes() == render(notice).tobytes()
