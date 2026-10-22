from collections.abc import Callable

import numpy
import pytest

from voicetotext.recording.notice import Notice
from voicetotext.recording.pill import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    DEFAULT_PILL_POSITION,
    NOTICE_CLOSE,
    Frame,
    Mode,
    PillPosition,
    notice_layout,
    notice_target,
    pill_box,
    pill_origin,
    render,
    render_notice,
    target_size,
)

_WORK_AREA = (0, 0, 1920, 1040)


class FakeRoot:
    """Tk's ``after`` with a clock the test moves by hand."""

    def __init__(self) -> None:
        self.pending: dict[str, tuple[int, Callable[[], None]]] = {}
        self.now = 100.0
        self._count = 0
        # Set while the fake window procedure runs: a call into Tk from
        # there crashes the real interpreter.
        self.in_window_procedure = False

    def after(self, ms: int, callback: Callable[[], None]) -> str:
        assert not self.in_window_procedure, "Tk was called from the window procedure"
        self._count += 1
        key = f"after#{self._count}"
        self.pending[key] = (ms, callback)
        return key

    def after_cancel(self, key: str) -> None:
        assert not self.in_window_procedure, "Tk was called from the window procedure"
        self.pending.pop(key, None)

    def fire(self, ms: int) -> None:
        self.now += ms / 1000
        for key, (delay, callback) in list(self.pending.items()):
            if delay <= ms and self.pending.pop(key, None) is not None:
                callback()

    def run(self, seconds: float) -> None:
        """Let the notice's frame timer run for ``seconds``."""
        for _ in range(round(seconds / 0.016)):
            self.fire(16)


class FakeNoticeWindow:
    """Records what the notice asks of Windows; never shows anything."""

    def __init__(self, root: FakeRoot) -> None:
        self._root = root
        self.size = (0, 0)
        self.shown: list[tuple[int, int, int, int]] = []
        self.moves: list[tuple[int, int]] = []
        self.frames: list[bytes] = []
        self.hidden = 0
        self.hand = False
        self.cursor: tuple[int, int] | None = None
        self._handler = None

    def work_area_for_foreground(self) -> tuple[int, int, int, int]:
        return _WORK_AREA

    def resize(self, width: int, height: int) -> None:
        self.size = (width, height)

    def show_at(self, x: int, y: int) -> None:
        self.shown.append((x, y, *self.size))

    def move_to(self, x: int, y: int) -> None:
        self.moves.append((x, y))

    @property
    def top(self) -> int:
        return self.moves[-1][1] if self.moves else self.shown[-1][1]

    def push_frame(self, pixels: bytes) -> None:
        assert len(pixels) == self.size[0] * self.size[1] * 4
        self.frames.append(pixels)

    def hide(self) -> None:
        self.hidden += 1

    def dispose(self) -> None:
        pass

    def set_mouse_handler(self, handler) -> None:
        self._handler = handler

    def cursor_position(self) -> tuple[int, int] | None:
        return self.cursor

    def set_hand_cursor(self, hand: bool) -> None:
        self.hand = hand

    def press(self, x: float, y: float) -> None:
        self._in_window_procedure(True, x, y)

    def release(self, x: float, y: float) -> None:
        self._in_window_procedure(False, x, y)

    def click(self, point: tuple[float, float]) -> None:
        self.press(*point)
        self.release(*point)
        # The notice's own timer acts on the click.
        self._root.fire(30)

    def _in_window_procedure(self, is_down: bool, x: float, y: float) -> None:
        self._root.in_window_procedure = True
        try:
            self._handler(is_down, round(x), round(y))
        finally:
            self._root.in_window_procedure = False


def _centre(box: tuple[float, float, float, float]) -> tuple[float, float]:
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


class Setup:
    def __init__(self, position: PillPosition = DEFAULT_PILL_POSITION) -> None:
        self.root = FakeRoot()
        self.window = FakeNoticeWindow(self.root)
        self.pressed: list[str] = []
        self.pill_up = False
        self.position = position
        self.notice = Notice(
            self.root, lambda: self.window,
            load_position=lambda: position,
            pill_showing=lambda: self.pill_up,
            clock=lambda: self.root.now,
        )

    def pill_box_on_screen(self) -> tuple[float, float, float, float]:
        pill_x, pill_y = pill_origin(self.position, _WORK_AREA)
        left, top, right, bottom = pill_box(*target_size(Mode.RECORDING))
        return pill_x + left, pill_y + top, pill_x + right, pill_y + bottom

    def action(self, name: str) -> Callable[[], None]:
        return lambda: self.pressed.append(name)

    def button(self, index: int) -> tuple[float, float]:
        return _centre(self.notice.layout.buttons[index])

    def close(self) -> tuple[float, float]:
        cx, cy, _radius = self.notice.layout.close
        return cx, cy


@pytest.fixture()
def setup() -> Setup:
    return Setup()


def test_a_one_line_notice_is_exactly_as_tall_as_the_pill_and_opens_where_the_pill_was(
    setup,
) -> None:
    setup.notice.show(
        "Transcription failed.",
        [("Retry", setup.action("retry")), ("Open history", setup.action("history"))],
        warning=True,
    )

    x, y, width, height = setup.window.shown[-1]
    left, top, right, bottom = setup.pill_box_on_screen()
    # A transparent pixel above and below keeps the border from being cut off.
    assert height == target_size(Mode.RECORDING)[1] + 2
    assert x + width / 2 == pytest.approx((left + right) / 2, abs=1)
    assert y + height / 2 == pytest.approx((top + bottom) / 2, abs=1)
    assert setup.notice.visible
    assert setup.notice.message == "Transcription failed."
    assert setup.notice.layout.labels == ("Retry", "Open history")


def test_a_notice_opens_where_the_pill_was_dragged() -> None:
    setup = Setup(position=(0.25, 0.4))

    setup.notice.show("Recording cancelled.", [("Transcribe anyway", setup.action("retry"))])

    x, y, width, height = setup.window.shown[-1]
    left, top, right, bottom = setup.pill_box_on_screen()
    assert x + width / 2 == pytest.approx((left + right) / 2, abs=1)
    assert y + height / 2 == pytest.approx((top + bottom) / 2, abs=1)


def test_it_fades_in_rather_than_appearing_at_once(setup) -> None:
    setup.notice.show("Recording cancelled.", [("Transcribe anyway", setup.action("retry"))])
    seen = [setup.notice.opacity]

    for _ in range(40):
        setup.root.run(0.016)
        seen.append(setup.notice.opacity)

    assert seen[0] == 0.0
    assert seen == sorted(seen)
    assert seen[-1] == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("position", [(0.0, 1.0), (1.0, 1.0)])
def test_a_notice_beside_a_pill_in_a_corner_stays_on_the_screen(position) -> None:
    setup = Setup(position=position)

    setup.notice.show(
        "No text box here, so it's on your clipboard.",
        [("Open history", setup.action("history"))],
        warning=True,
    )

    x, _y, width, _height = setup.window.shown[-1]
    assert 0 <= x and x + width <= 1920


def test_a_long_message_wraps_onto_more_lines_instead_of_running_off_the_screen() -> None:
    message = "Zoë's naïve café dictation ✓ 東京 " * 40

    layout = notice_layout(message, ["Retry"], warning=True)

    assert len(layout.lines) > 1
    assert layout.width < 640
    assert layout.height > target_size(Mode.RECORDING)[1] + 2
    assert " ".join(layout.lines).split() == message.split()


def test_a_single_word_too_long_for_a_line_is_cut_short() -> None:
    layout = notice_layout("x" * 5000, [], warning=False)

    assert len(layout.lines) == 1
    assert layout.lines[0].endswith("…")
    assert layout.width < 640


def test_an_empty_message_still_gets_a_close_cross() -> None:
    layout = notice_layout("", [], warning=False)

    assert layout.lines == ("",)
    assert layout.buttons == ()
    assert layout.close[2] > 0


def test_each_button_runs_its_own_action_and_puts_the_notice_away(setup) -> None:
    setup.notice.show(
        "Transcription failed.",
        [("Retry", setup.action("retry")), ("Open history", setup.action("history"))],
    )

    setup.window.click(setup.button(1))

    assert setup.pressed == ["history"]
    assert not setup.notice.visible
    assert setup.window.hidden == 1


def test_the_close_cross_puts_the_notice_away_without_running_any_action(setup) -> None:
    setup.notice.show("Transcription failed.", [("Retry", setup.action("retry"))])

    setup.window.click(setup.close())

    assert setup.pressed == []
    assert not setup.notice.visible


def test_a_press_that_slides_off_its_button_does_nothing(setup) -> None:
    setup.notice.show(
        "Transcription failed.",
        [("Retry", setup.action("retry")), ("Open history", setup.action("history"))],
    )

    setup.window.press(*setup.button(0))
    setup.window.release(*setup.button(1))
    setup.window.press(*setup.button(0))
    setup.window.release(*setup.close())
    setup.root.fire(30)

    assert setup.pressed == []
    assert setup.notice.visible


def test_a_click_on_the_message_itself_does_nothing(setup) -> None:
    setup.notice.show("Transcription failed.", [("Retry", setup.action("retry"))])
    layout = setup.notice.layout

    setup.window.click((layout.text_left + 4, layout.height / 2))

    assert setup.pressed == []
    assert setup.notice.visible


def test_a_second_notice_replaces_the_first(setup) -> None:
    setup.notice.show("Recording cancelled.", [("Transcribe anyway", setup.action("old"))])
    setup.notice.show(
        "No text box here, so it's on your clipboard.",
        [("Open history", setup.action("history"))],
    )

    setup.window.click(setup.button(0))

    assert setup.pressed == ["history"]
    assert len(setup.window.shown) == 2


def test_at_most_three_buttons_are_offered(setup) -> None:
    setup.notice.show("Choose", [(f"Option {n}", setup.action(str(n))) for n in range(5)])

    assert setup.notice.layout.labels == ("Option 0", "Option 1", "Option 2")


def test_a_notice_fades_away_on_its_own_after_ten_seconds(setup) -> None:
    setup.notice.show("Recording cancelled.", [("Transcribe anyway", setup.action("retry"))])

    setup.root.run(9.9)
    assert setup.notice.opacity == pytest.approx(1.0, abs=0.02)
    fading = []
    for _ in range(60):
        setup.root.run(0.016)
        if setup.notice.visible:
            fading.append(setup.notice.opacity)

    assert fading == sorted(fading, reverse=True)
    assert not setup.notice.visible
    assert setup.window.hidden == 1
    assert setup.root.pending == {}


def test_pointing_at_a_notice_keeps_it_from_fading(setup) -> None:
    setup.notice.show("Recording cancelled.", [("Transcribe anyway", setup.action("retry"))])
    setup.window.cursor = (10, 10)

    setup.root.run(20.0)

    assert setup.notice.visible
    assert setup.notice.opacity == pytest.approx(1.0, abs=0.02)
    setup.window.cursor = None
    setup.root.run(4.0)
    assert not setup.notice.visible


class TestWhileThePillIsUp:
    def _tucked(self, setup: Setup) -> None:
        setup.notice.show("Recording cancelled.", [("Transcribe anyway", setup.action("retry"))])
        setup.root.run(1.0)
        setup.pill_up = True
        setup.root.run(1.0)

    def test_starting_a_recording_slides_it_up_half_behind_the_pill(self, setup) -> None:
        self._tucked(setup)

        _left, top, _right, _bottom = setup.pill_box_on_screen()
        height = setup.window.size[1]
        assert setup.window.top < top < setup.window.top + height
        assert setup.notice.opacity < 1.0

    def test_only_the_strip_past_the_pills_edge_is_drawn(self, setup) -> None:
        self._tucked(setup)

        width, height = setup.window.size
        pixels = numpy.frombuffer(setup.window.frames[-1], numpy.uint8).reshape(height, width, 4)
        _left, top, _right, _bottom = setup.pill_box_on_screen()
        cut = round(top) - setup.window.top
        assert 0 < cut < height
        assert pixels[:cut, :, 3].any()
        assert not pixels[cut:].any()

    def test_it_only_moves_so_the_pill_stays_in_front(self, setup) -> None:
        self._tucked(setup)

        assert len(setup.window.shown) == 1

    def test_pointing_at_the_part_that_shows_brings_it_out_above_the_pill(self, setup) -> None:
        self._tucked(setup)

        setup.window.cursor = (20, 4)
        setup.root.run(0.6)

        _left, top, _right, _bottom = setup.pill_box_on_screen()
        assert setup.window.top + setup.window.size[1] <= top
        assert setup.notice.opacity == pytest.approx(1.0, abs=0.02)

    def test_pointing_through_the_pill_at_the_hidden_part_does_nothing(self, setup) -> None:
        self._tucked(setup)
        tucked_top = setup.window.top

        setup.window.cursor = (setup.window.size[0] // 2, setup.window.size[1] - 4)
        setup.root.run(0.6)

        assert setup.window.top == tucked_top

    def test_left_alone_behind_the_pill_it_fades_away_sooner(self, setup) -> None:
        self._tucked(setup)

        setup.root.run(3.5)

        assert not setup.notice.visible

    def test_a_notice_raised_while_the_pill_is_up_starts_tucked(self, setup) -> None:
        setup.pill_up = True

        setup.notice.show("Writing model is not ready.", [])

        _x, y, _width, height = setup.window.shown[-1]
        _left, top, _right, _bottom = setup.pill_box_on_screen()
        assert y < top < y + height

    def test_by_a_pill_at_the_top_of_the_screen_it_tucks_below_instead(self) -> None:
        setup = Setup(position=(0.5, 0.0))
        self._tucked(setup)

        _left, _top, _right, bottom = setup.pill_box_on_screen()
        assert setup.window.top < bottom < setup.window.top + setup.window.size[1]
        assert setup.window.top >= 0

    def test_it_settles_back_where_the_pill_was_once_the_pill_has_gone(self, setup) -> None:
        setup.notice.show("Recording cancelled.", [])
        setup.root.run(1.0)
        resting = setup.window.top
        setup.pill_up = True
        setup.root.run(0.5)

        setup.pill_up = False
        setup.root.run(1.0)

        assert setup.window.top == resting


def test_hiding_a_notice_that_is_not_showing_does_nothing(setup) -> None:
    setup.notice.hide()
    setup.notice.show("Transcription failed.", [])
    setup.notice.hide()
    setup.notice.hide()

    assert setup.window.hidden == 1


def test_hovering_a_button_lights_it_and_shows_the_hand_cursor(setup) -> None:
    setup.notice.show(
        "Transcription failed.",
        [("Retry", setup.action("retry")), ("Open history", setup.action("history"))],
    )
    setup.root.run(1.0)
    resting = setup.window.frames[-1]

    x, y = setup.button(1)
    setup.window.cursor = (round(x), round(y))
    setup.root.fire(30)
    assert setup.window.hand
    assert setup.window.frames[-1] != resting

    setup.window.cursor = (-50, -50)
    setup.root.fire(30)
    assert not setup.window.hand
    assert setup.window.frames[-1] == resting


def test_a_notice_is_drawn_in_the_pills_own_colour() -> None:
    width, height = target_size(Mode.RECORDING)
    left, top, _right, _bottom = pill_box(width, height)
    pill = render(Frame(width, height, 1.0, 0.0, 1.0)).getpixel((round(left + 60), round(top + 3)))

    notice = render_notice(notice_layout("Transcription failed.", ["Retry"], warning=True))

    # Bare capsule, right of the left curve and above the text.
    assert notice.getpixel((26, 4)) == pill


def test_the_close_cross_is_told_apart_from_the_buttons() -> None:
    layout = notice_layout("Transcription failed.", ["Retry"], warning=False)
    cx, cy, _radius = layout.close

    assert notice_target(layout, cx, cy) == NOTICE_CLOSE
    assert notice_target(layout, *_centre(layout.buttons[0])) == 0
    assert notice_target(layout, 0, 0) is None
