from collections.abc import Callable

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
        for key, (delay, callback) in list(self.pending.items()):
            if delay <= ms and self.pending.pop(key, None) is not None:
                callback()


class FakeNoticeWindow:
    """Records what the notice asks of Windows; never shows anything."""

    def __init__(self, root: FakeRoot) -> None:
        self._root = root
        self.size = (0, 0)
        self.shown: list[tuple[int, int, int, int]] = []
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
        self.notice = Notice(self.root, lambda: self.window, load_position=lambda: position)

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


def test_a_one_line_notice_is_exactly_as_tall_as_the_pill_and_sits_centred_just_above_it(
    setup,
) -> None:
    setup.notice.show(
        "Transcription failed.",
        [("Retry", setup.action("retry")), ("Open history", setup.action("history"))],
        warning=True,
    )

    x, y, width, height = setup.window.shown[-1]
    pill_x, pill_y = pill_origin(DEFAULT_PILL_POSITION, _WORK_AREA)
    # A transparent pixel above and below keeps the border from being cut off.
    assert height == target_size(Mode.RECORDING)[1] + 2
    assert x + width / 2 == pytest.approx(pill_x + CANVAS_WIDTH / 2, abs=1)
    assert y + height == pill_y - 8
    assert setup.notice.visible
    assert setup.notice.message == "Transcription failed."
    assert setup.notice.layout.labels == ("Retry", "Open history")


def test_a_notice_follows_the_pill_to_wherever_it_was_dragged() -> None:
    setup = Setup(position=(0.25, 0.4))

    setup.notice.show("Recording cancelled.", [("Transcribe anyway", setup.action("retry"))])

    x, y, width, height = setup.window.shown[-1]
    pill_x, pill_y = pill_origin((0.25, 0.4), _WORK_AREA)
    assert x + width / 2 == pytest.approx(pill_x + CANVAS_WIDTH / 2, abs=1)
    assert y + height == pill_y - 8


def test_a_notice_for_a_pill_at_the_top_of_the_screen_sits_below_it() -> None:
    setup = Setup(position=(0.5, 0.0))

    setup.notice.show("Recording cancelled.", [("Transcribe anyway", setup.action("retry"))])

    _x, y, _width, _height = setup.window.shown[-1]
    assert y == CANVAS_HEIGHT + 8


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


def test_a_notice_hides_itself_after_ten_seconds(setup) -> None:
    setup.notice.show("Recording cancelled.", [("Transcribe anyway", setup.action("retry"))])

    setup.root.fire(9_999)
    assert setup.notice.visible
    setup.root.fire(10_000)

    assert not setup.notice.visible
    assert setup.window.hidden == 1
    assert setup.root.pending == {}


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
