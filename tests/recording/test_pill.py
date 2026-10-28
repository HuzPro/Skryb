import itertools
import math
import sys

import numpy
import pytest

from skryb.recording.pill import (
    BOTTOM_MARGIN,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_PILL_POSITION,
    Button,
    Frame,
    Mode,
    Spring,
    VoiceHistory,
    VoiceLevel,
    button_centres,
    countdown_text,
    hit_test,
    is_settled_label,
    pill_box,
    pill_origin,
    pill_position,
    render,
    settled_label_frame,
    shimmer_position,
    shimmered,
    target_size,
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


def _settled_pixels():
    from skryb.recording.overlay import premultiplied_array

    return premultiplied_array(render(settled_label_frame()))


def _shimmer_at(position: float) -> Frame:
    return Frame(**{**settled_label_frame().__dict__, "shimmer": position})


class TestShimmer:
    def test_each_pass_starts_at_the_left_and_ends_at_the_right(self):
        assert shimmer_position(0.0) == 0.0
        assert shimmer_position(0.8999) > 0.99

    def test_it_repeats_while_the_transcription_runs(self):
        assert shimmer_position(0.9 + 0.2) == pytest.approx(shimmer_position(0.2))

    def test_it_eases_in_and_out_of_each_pass(self):
        start = shimmer_position(0.05) - shimmer_position(0.0)
        middle = shimmer_position(0.475) - shimmer_position(0.425)

        assert start < middle / 3

    def test_a_clock_that_ran_backwards_counts_as_just_started(self):
        assert shimmer_position(-3.0) == 0.0

    def test_the_band_lifts_the_fill_under_it_and_leaves_the_rest_alone(self):
        base = _settled_pixels()
        width, height = settled_label_frame().width, settled_label_frame().height
        left, top, right, bottom = pill_box(width, height)
        lit = shimmered(base, _shimmer_at(0.5))
        centre_x, far_x = round((left + right) / 2), round(left + 4)
        row = round(bottom - 4)

        assert lit[row, centre_x, :3].sum() > base[row, centre_x, :3].sum() + 60
        assert (lit[:, far_x] == base[:, far_x]).all()

    @pytest.mark.parametrize("position", [0.0, 0.17, 0.5, 0.83, 1.0])
    def test_nothing_gets_darker_so_the_white_label_reads_the_same(self, position):
        base = _settled_pixels()

        assert (shimmered(base, _shimmer_at(position)) >= base).all()

    def test_no_band_outside_transcribing(self):
        base = _settled_pixels()

        assert shimmered(base, settled_label_frame()) is base
        assert shimmered(base, Frame(**{**_shimmer_at(0.5).__dict__, "label": 0.0})) is base

    def test_the_cached_pill_is_only_used_once_it_has_fully_settled(self):
        width, height = target_size(Mode.TRANSCRIBING)

        assert is_settled_label(Frame(width, height, 1.0, 0.0, 1.0))
        assert not is_settled_label(Frame(width + 2, height, 1.0, 0.0, 1.0))
        assert not is_settled_label(Frame(width, height, 0.9, 0.0, 1.0))
        assert not is_settled_label(Frame(width, height, 1.0, 0.2, 1.0))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows fonts")
def test_the_pill_draws_its_text_in_segoe_ui_semibold() -> None:
    from skryb.recording.pill import _font

    assert _font(36).getname() == ("Segoe UI", "Semibold")


class TestOnePill:
    def test_the_transcribing_bar_takes_the_recording_pills_exact_size_and_place(self):
        recording, transcribing = target_size(Mode.RECORDING), target_size(Mode.TRANSCRIBING)

        assert transcribing == recording
        assert pill_box(*transcribing) == pill_box(*recording)

    def test_the_transcribing_bar_keeps_the_recording_pills_colour(self):
        width, height = target_size(Mode.RECORDING)
        left, top, _right, _bottom = pill_box(width, height)
        # Bare pill in both: above the buttons, the waveform and the label.
        point = (round(left + 60), round(top + 3))

        recording = render(_recording_frame()).getpixel(point)
        transcribing = render(Frame(width, height, 1.0, 0.0, 1.0)).getpixel(point)

        assert transcribing == recording


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
        image = render(_recording_frame(levels=(0.8,) * 13, scroll=0.4))

        assert image.size == (CANVAS_WIDTH, CANVAS_HEIGHT)
        assert image.getpixel((0, 0))[3] == 0
        assert image.getpixel((CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1))[3] == 0

    def test_the_waveform_moves_with_the_voice(self):
        quiet = render(_recording_frame(levels=(0.0,) * 13))
        loud = render(_recording_frame(levels=(1.0,) * 13))

        assert quiet.tobytes() != loud.tobytes()

    def test_the_row_slides_by_fractions_of_a_slot_between_slices(self):
        levels = tuple(i / 12 for i in range(13))
        frames = {render(_recording_frame(levels=levels, scroll=i / 8)).tobytes() for i in range(8)}

        assert len(frames) == 8

    def test_a_paused_pill_looks_different_from_a_recording_one(self):
        recording = render(_recording_frame(levels=(0.5,) * 13))
        paused = render(_recording_frame(levels=(0.5,) * 13, paused=True))

        assert recording.tobytes() != paused.tobytes()

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


def _db(value: float) -> float:
    return 10 ** (value / 20)


# Speech the way a microphone hears it: syllables with short gaps over the
# room's own noise, read in 64 ms blocks like the recorder delivers them.
_BLOCK = 1024 / 16000


def _speak(level: VoiceLevel, voice_db: float, seconds: float, room_db: float = -62.0) -> list[float]:
    values = []
    for i in range(round(seconds / _BLOCK)):
        talking = i % 7 not in (5, 6)
        wobble = (i * 37 % 11 - 5) * 0.6
        values.append(level.update(_db(voice_db + wobble if talking else room_db), _BLOCK))
    return values


def _talking(values: list[float]) -> list[float]:
    return [v for i, v in enumerate(values) if i % 7 not in (5, 6)]


class TestVoiceLevel:
    def test_quiet_speech_moves_the_bars_clearly(self):
        values = _talking(_speak(VoiceLevel(), -45.0, 6.0))[20:]

        assert numpy.median(values) > 0.4

    def test_speech_from_across_the_room_fills_the_bars_like_speech_up_close(self):
        near = numpy.median(_talking(_speak(VoiceLevel(), -24.0, 6.0))[20:])
        far = numpy.median(_talking(_speak(VoiceLevel(), -42.0, 6.0, room_db=-70.0))[20:])

        assert far == pytest.approx(near, abs=0.15)

    def test_louder_syllables_draw_taller_bars_than_quieter_ones(self):
        level = VoiceLevel()
        _speak(level, -32.0, 4.0)

        soft = level.update(_db(-40.0), _BLOCK)
        strong = level.update(_db(-26.0), _BLOCK)

        assert strong > soft + 0.2

    def test_room_noise_alone_draws_flat(self):
        level = VoiceLevel()
        values = [level.update(_db(-60.0 + (i % 5) * 0.8), _BLOCK) for i in range(200)]

        assert max(values[10:]) == 0.0

    def test_one_cough_does_not_flatten_the_next_sentence_for_long(self):
        level = VoiceLevel()
        _speak(level, -32.0, 3.0)
        level.update(_db(-3.0), _BLOCK)

        later = _talking(_speak(level, -32.0, 4.0))[-15:]

        assert numpy.median(later) > 0.45

    def test_a_sudden_shout_never_overflows_the_bars(self):
        level = VoiceLevel()
        _speak(level, -40.0, 3.0)

        assert level.update(0.9, _BLOCK) == 1.0

    @pytest.mark.parametrize("rms", [math.nan, -1.0, math.inf, 0.0])
    def test_bad_readings_never_break_it(self, rms):
        level = VoiceLevel()

        value = level.update(rms, _BLOCK)

        assert 0.0 <= value <= 1.0
        assert 0.0 <= level.update(0.05, _BLOCK) <= 1.0


class TestVoiceHistory:
    def test_the_newest_slice_fills_while_you_speak(self):
        history = VoiceHistory(13)
        for _ in range(40):
            history.update(_db(-60.0), 1 / 240)
        history.update(_db(-25.0), 1 / 240)

        assert history.levels()[-1] > 0.5
        assert history.levels()[:-1] == (0.0,) * 12

    def test_each_slice_moves_one_slot_left_as_the_next_begins(self):
        history = VoiceHistory(13)
        history.update(_db(-60.0), 0.01)
        history.update(_db(-25.0), 0.01)
        newest = history.levels()[-1]

        for _ in range(10):
            history.update(_db(-60.0), 0.0075)

        assert history.levels()[-2] == newest

    def test_the_row_scrolls_smoothly_and_wraps_at_each_slice(self):
        history = VoiceHistory(13)
        scrolls = []
        for _ in range(60):
            history.update(_db(-40.0), 1 / 240)
            scrolls.append(history.scroll())

        assert all(0.0 <= value < 1.0 for value in scrolls)
        assert sum(after < before for before, after in itertools.pairwise(scrolls)) == 3

    def test_keeps_a_fixed_number_of_slices_however_long_you_talk(self):
        history = VoiceHistory(13)
        for _ in range(5000):
            history.update(_db(-30.0), 1 / 240)

        assert len(history.levels()) == 13

    def test_a_new_recording_starts_flat_but_remembers_the_speaker(self):
        history = VoiceHistory(13)
        for _ in range(400):
            history.update(_db(-45.0 if _ % 3 else -62.0), _BLOCK)

        history.clear()

        assert history.levels() == (0.0,) * 13
        assert history.scroll() == 0.0
        history.update(_db(-45.0), _BLOCK)
        assert history.levels()[-1] > 0.4


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
        plain = _recording_frame(levels=(1.0,) * 13)
        silent = _recording_frame(
            levels=(1.0,) * 13, silent_microphone="Maono PD300X Microphone",
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
        plain = render(_recording_frame(levels=(1.0,) * 13))
        counting = render(_recording_frame(levels=(1.0,) * 13, countdown="0:30"))

        assert _amber_pixels(_recording_frame(countdown="0:30")) > 20
        assert plain.tobytes() != counting.tobytes()

    def test_a_silent_microphone_outranks_the_countdown(self):
        both = _recording_frame(silent_microphone="Blue Yeti", countdown="0:30")
        notice = _recording_frame(silent_microphone="Blue Yeti")

        assert render(both).tobytes() == render(notice).tobytes()
