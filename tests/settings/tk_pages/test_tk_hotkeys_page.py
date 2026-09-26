import pytest

from voicetotext.hotkeys.bindings import Binding, BindingAction, BindingMode
from voicetotext.settings.signal import Signal
from voicetotext.settings.tk_pages.hotkeys_page import TkHotkeysPage

_TOGGLE = BindingMode.TOGGLE
_HOLD = BindingMode.HOLD
_PRESS = BindingMode.PRESS
_DICTATE = BindingAction.DICTATE


class FakeCaptureBridge:
    def __init__(self) -> None:
        self.key_pressed = Signal()
        self.running = 0

    def start(self) -> None:
        self.running += 1

    def stop(self) -> None:
        self.running -= 1

    def emit_key(self, name: str) -> None:
        self.key_pressed.emit(name)


@pytest.fixture
def bridge():
    return FakeCaptureBridge()


@pytest.fixture
def page(tk_root, bridge):
    widget = TkHotkeysPage(tk_root, capture_bridge=bridge)
    yield widget
    widget.destroy()


_EVERY_KIND = (
    Binding("ctrl+shift+alt+m", _TOGGLE, _DICTATE),
    Binding("mouse4", _HOLD, _DICTATE),
    Binding("ctrl+f13", _TOGGLE, _DICTATE),
    Binding("ctrl+alt+e", _HOLD, BindingAction.EDIT_SELECTION),
    Binding("ctrl+alt+v", _PRESS, BindingAction.PASTE_LAST),
    Binding("mouse5", _PRESS, BindingAction.COPY_LAST),
)


def test_three_dictation_shortcuts_and_the_three_optional_ones_round_trip(page):
    page.set_bindings(_EVERY_KIND)
    page.set_double_tap_locks_hold(False)

    assert page.bindings() == _EVERY_KIND
    assert page.double_tap_locks_hold() is False
    assert len(page.dictation_rows()) == 3
    assert page.validation_error() == ""


def test_loading_fewer_shortcuts_drops_the_extra_rows_and_blanks_the_optional_ones(page):
    page.set_bindings(_EVERY_KIND)

    page.set_bindings((Binding("f12", _HOLD, _DICTATE),))

    assert page.bindings() == (Binding("f12", _HOLD, _DICTATE),)
    assert len(page.dictation_rows()) == 1
    assert page.edit_row().capture.combination() == ""


def test_each_optional_shortcut_sits_under_its_own_heading(page):
    for row, heading in (
        (page.edit_row(), "Edit selected text by voice"),
        (page.paste_row(), "Paste last dictation"),
        (page.copy_row(), "Copy last dictation"),
    ):
        block = row.master.pack_slaves()
        assert block[0].cget("text") == heading
        assert block.index(row) > 0


def test_a_new_page_starts_with_one_empty_dictation_row_that_cannot_be_removed(page):
    only = page.dictation_rows()[0]

    page.remove_dictation_row(only)

    assert page.dictation_rows() == [only]
    assert only.remove_button is not None
    assert only.remove_button.instate(["disabled"])
    assert page.double_tap_locks_hold() is True


def test_an_added_row_can_be_removed_again_down_to_the_last_one(page):
    page.set_bindings((Binding("f12", _TOGGLE, _DICTATE),))
    added = page.add_dictation_row()
    added.capture.set_combination("mouse4")
    added.set_mode(_HOLD)

    assert page.bindings() == (
        Binding("f12", _TOGGLE, _DICTATE), Binding("mouse4", _HOLD, _DICTATE),
    )
    page.remove_dictation_row(added)

    assert page.bindings() == (Binding("f12", _TOGGLE, _DICTATE),)
    remaining = page.dictation_rows()[0]
    assert remaining.remove_button is not None
    assert remaining.remove_button.instate(["disabled"])


def test_blank_rows_are_left_out_and_blank_optional_shortcuts_stay_off(page):
    page.set_bindings((Binding("f12", _TOGGLE, _DICTATE),))
    page.add_dictation_row()

    assert page.bindings() == (Binding("f12", _TOGGLE, _DICTATE),)


def test_without_any_dictation_shortcut_save_is_refused(page):
    page.paste_row().capture.set_combination("ctrl+alt+v")

    assert "at least one dictation shortcut" in page.validation_error()


def test_a_shortcut_that_starts_another_is_refused_with_both_names(page):
    page.set_bindings((
        Binding("ctrl+alt", _HOLD, _DICTATE),
        Binding("ctrl+alt+m", _TOGGLE, _DICTATE),
    ))

    message = page.validation_error()

    assert "ctrl+alt+m" in message and "ctrl+alt" in message


def test_an_invalid_typed_shortcut_is_reported_before_save(page):
    page.dictation_rows()[0].capture.set_typed_text("ctrl+ü")

    message = page.validation_error()

    assert "ctrl+ü" in message
    assert "Unknown key" in message


def test_recording_one_shortcut_stops_recording_any_other(page, bridge):
    events: list[bool] = []
    page.recording_state_changed.connect(events.append)
    first = page.dictation_rows()[0].capture
    second = page.paste_row().capture

    first.start_recording()
    second.start_recording()
    bridge.emit_key("ctrl")
    bridge.emit_key("mouse5")
    second.stop_recording()

    assert first.is_recording() is False
    assert second.combination() == "ctrl+mouse5"
    assert first.combination() == ""
    assert events == [True, True, False]
    assert bridge.running == 0


def test_reloading_the_shortcuts_many_times_leaves_only_the_visible_boxes_listening_to_the_hooks(
    tk_root, bridge,
):
    scheduled: list = []
    page = TkHotkeysPage(tk_root, capture_bridge=bridge, schedule=scheduled.append)
    try:
        for _ in range(200):
            page.set_bindings(_EVERY_KIND)
        page.remove_dictation_row(page.dictation_rows()[-1])

        bridge.emit_key("f9")

        assert len(page.dictation_rows()) == 2
        assert len(scheduled) == 2 + 3
    finally:
        page.destroy()
