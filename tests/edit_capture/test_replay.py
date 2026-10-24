"""Tests for ``EditReplay``.

Replay is pure - no Win32, no I/O - so every test drives it through
``apply_paste``/``apply_key`` and asserts on the returned ``EditState``.
"""

import pytest

from skryb.edit_capture.replay import (
    VK_A,
    VK_BACK,
    VK_DELETE,
    VK_DOWN,
    VK_END,
    VK_ESCAPE,
    VK_HOME,
    VK_LEFT,
    VK_NEXT,
    VK_PRIOR,
    VK_PROCESSKEY,
    VK_RETURN,
    VK_RIGHT,
    VK_TAB,
    VK_UP,
    VK_V,
    VK_X,
    VK_Z,
    EditFlag,
    EditReplay,
    EditState,
    KeyStroke,
    PasteSpan,
)

pytestmark = pytest.mark.unit

# Punctuation and space arrive on OEM keys, whose codes sit above every
# virtual key the replay reacts to. Reusing the character's own ordinal
# would make "!" (0x21) arrive as Page Up.
VK_OEM_PUNCTUATION = 0xBA


def _pasted(text: str) -> EditReplay:
    replay = EditReplay()
    replay.apply_paste(text)
    return replay


def _type(replay: EditReplay, text: str) -> EditState:
    state = replay.snapshot()
    for char in text:
        vk = ord(char.upper()) if char.isalnum() else VK_OEM_PUNCTUATION
        state = replay.apply_key(KeyStroke(vk, char))
    return state


class TestTyping:
    def test_should_append_characters_typed_after_the_paste(self) -> None:
        replay = _pasted("hello")

        state = _type(replay, " there")

        assert state.text == "hello there"

    def test_should_insert_at_the_caret_after_moving_left(self) -> None:
        replay = _pasted("ac")
        replay.apply_key(KeyStroke(VK_LEFT, ""))

        state = _type(replay, "b")

        assert state.text == "abc"

    def test_should_ignore_characters_typed_with_control_held(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(ord("S"), "s", ctrl=True))

        assert state.text == "hello"

    def test_should_insert_the_character_an_altgr_stroke_types(self) -> None:
        replay = _pasted("price ")

        state = replay.apply_key(KeyStroke(ord("E"), "€", ctrl=True, alt=True))

        assert state.text == "price €"
        assert state.reliable is True

    def test_should_keep_treating_alt_backspace_as_an_undo(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_BACK, "", ctrl=True, alt=True))

        assert state.text == "hello"
        assert EditFlag.UNDO in state.flags


class TestDeletion:
    def test_should_delete_the_character_before_the_caret(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_BACK, ""))

        assert state.text == "hell"

    def test_should_flag_caret_before_span_when_backspacing_at_the_start(self) -> None:
        replay = _pasted("a")
        replay.apply_key(KeyStroke(VK_LEFT, ""))

        state = replay.apply_key(KeyStroke(VK_BACK, ""))

        assert state.text == "a"
        assert EditFlag.CARET_BEFORE_SPAN in state.flags

    def test_should_delete_the_word_before_the_caret_with_control(self) -> None:
        replay = _pasted("hello world")

        state = replay.apply_key(KeyStroke(VK_BACK, "", ctrl=True))

        assert state.text == "hello "

    def test_should_delete_trailing_space_and_word_with_control(self) -> None:
        replay = _pasted("hello world ")

        state = replay.apply_key(KeyStroke(VK_BACK, "", ctrl=True))

        assert state.text == "hello "

    def test_should_delete_the_word_and_its_space_after_the_caret(self) -> None:
        replay = _pasted("hello world")
        replay.apply_key(KeyStroke(VK_HOME, "", ctrl=True))

        state = replay.apply_key(KeyStroke(VK_DELETE, "", ctrl=True))

        assert state.text == "world"

    def test_should_delete_the_selection_on_control_delete(self) -> None:
        replay = _pasted("hello world")
        for _ in range(len("world")):
            replay.apply_key(KeyStroke(VK_LEFT, "", shift=True))

        state = replay.apply_key(KeyStroke(VK_DELETE, "", ctrl=True))

        assert state.text == "hello "

    def test_should_delete_the_character_at_the_caret(self) -> None:
        replay = _pasted("abc")
        replay.apply_key(KeyStroke(VK_LEFT, ""))

        state = replay.apply_key(KeyStroke(VK_DELETE, ""))

        assert state.text == "ab"

    def test_should_flag_caret_after_span_when_deleting_at_the_end(self) -> None:
        replay = _pasted("abc")

        state = replay.apply_key(KeyStroke(VK_DELETE, ""))

        assert state.text == "abc"
        assert EditFlag.CARET_AFTER_SPAN in state.flags


class TestSelection:
    def test_should_replace_a_shift_left_selection_with_the_typed_character(
        self,
    ) -> None:
        replay = _pasted("hello")
        replay.apply_key(KeyStroke(VK_LEFT, "", shift=True))

        state = _type(replay, "p")

        assert state.text == "hellp"

    def test_should_replace_the_whole_buffer_after_select_all(self) -> None:
        replay = _pasted("hello")
        replay.apply_key(KeyStroke(VK_A, "", ctrl=True))

        state = _type(replay, "bye")

        assert state.text == "bye"
        assert EditFlag.SELECT_ALL in state.flags

    def test_should_cut_the_selection_with_control_x(self) -> None:
        replay = _pasted("hello")
        replay.apply_key(KeyStroke(VK_LEFT, "", shift=True))
        replay.apply_key(KeyStroke(VK_LEFT, "", shift=True))

        state = replay.apply_key(KeyStroke(VK_X, "", ctrl=True))

        assert state.text == "hel"

    def test_should_collapse_the_selection_when_moving_without_shift(self) -> None:
        replay = _pasted("hello")
        replay.apply_key(KeyStroke(VK_LEFT, "", shift=True))
        replay.apply_key(KeyStroke(VK_LEFT, ""))

        state = _type(replay, "X")

        assert state.text == "hellXo"

    def test_should_collapse_to_the_right_edge_when_right_ends_a_selection(
        self,
    ) -> None:
        replay = _pasted("hello world")
        for _ in range(len("world")):
            replay.apply_key(KeyStroke(VK_LEFT, "", shift=True))
        replay.apply_key(KeyStroke(VK_RIGHT, ""))

        state = _type(replay, "Z")

        assert state.text == "hello worldZ"

    def test_should_collapse_to_the_left_edge_when_control_left_ends_a_selection(
        self,
    ) -> None:
        replay = _pasted("hello world")
        for _ in range(len("world")):
            replay.apply_key(KeyStroke(VK_LEFT, "", shift=True))
        replay.apply_key(KeyStroke(VK_LEFT, "", ctrl=True))

        state = _type(replay, "Z")

        assert state.text == "hello Zworld"

    def test_should_still_extend_a_selection_when_shift_is_held(self) -> None:
        replay = _pasted("hello")
        replay.apply_key(KeyStroke(VK_LEFT, "", shift=True))
        replay.apply_key(KeyStroke(VK_LEFT, "", shift=True))

        state = _type(replay, "X")

        assert state.text == "helX"


class TestCaretMotion:
    def test_should_move_one_word_left_with_control(self) -> None:
        replay = _pasted("hello world")
        replay.apply_key(KeyStroke(VK_LEFT, "", ctrl=True))

        state = _type(replay, "X")

        assert state.text == "hello Xworld"

    def test_should_move_one_word_right_with_control(self) -> None:
        replay = _pasted("hello world")
        replay.apply_key(KeyStroke(VK_HOME, "", ctrl=True))
        replay.apply_key(KeyStroke(VK_RIGHT, "", ctrl=True))

        state = _type(replay, "X")

        assert state.text == "hello Xworld"

    def test_should_flag_caret_before_span_when_left_is_clamped(self) -> None:
        replay = _pasted("a")
        replay.apply_key(KeyStroke(VK_LEFT, ""))

        state = replay.apply_key(KeyStroke(VK_LEFT, ""))

        assert EditFlag.CARET_BEFORE_SPAN in state.flags

    def test_should_flag_caret_after_span_when_right_is_clamped(self) -> None:
        replay = _pasted("a")

        state = replay.apply_key(KeyStroke(VK_RIGHT, ""))

        assert EditFlag.CARET_AFTER_SPAN in state.flags

    def test_should_move_home_to_the_start_of_the_current_line(self) -> None:
        replay = _pasted("one")
        replay.apply_key(KeyStroke(VK_RETURN, ""))
        _type(replay, "two")
        replay.apply_key(KeyStroke(VK_HOME, ""))

        state = _type(replay, "X")

        assert state.text == "one\nXtwo"

    def test_should_move_end_to_the_end_of_the_current_line(self) -> None:
        replay = _pasted("one")
        replay.apply_key(KeyStroke(VK_RETURN, ""))
        _type(replay, "two")
        replay.apply_key(KeyStroke(VK_HOME, ""))
        replay.apply_key(KeyStroke(VK_LEFT, ""))
        replay.apply_key(KeyStroke(VK_END, ""))

        state = _type(replay, "X")

        assert state.text == "oneX\ntwo"

    def test_should_flag_vertical_arrows_and_leave_the_caret_alone(self) -> None:
        replay = _pasted("hello")

        replay.apply_key(KeyStroke(VK_UP, ""))
        state = _type(replay, "!")

        assert state.text == "hello!"
        assert EditFlag.VERTICAL_ARROW in state.flags

    def test_should_flag_page_keys_as_vertical_arrows(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_NEXT, ""))

        assert EditFlag.VERTICAL_ARROW in state.flags


class TestCommit:
    def test_should_return_the_text_before_the_newline_as_committed(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_RETURN, ""))

        assert state.committed is True
        assert state.text == "hello"

    def test_should_keep_editing_after_a_commit(self) -> None:
        replay = _pasted("hello")
        replay.apply_key(KeyStroke(VK_RETURN, ""))

        state = _type(replay, "again")

        assert state.committed is False
        assert state.text == "hello\nagain"

    def test_should_commit_the_whole_buffer_at_the_second_enter(self) -> None:
        replay = _pasted("one")
        replay.apply_key(KeyStroke(VK_RETURN, ""))
        _type(replay, "two")

        state = replay.apply_key(KeyStroke(VK_RETURN, ""))

        assert state.text == "one\ntwo"

    def test_should_not_commit_a_snapshot(self) -> None:
        replay = _pasted("hello")

        assert replay.snapshot() == replay.snapshot()
        assert replay.snapshot().committed is False


class TestSpans:
    def test_should_report_one_span_per_paste(self) -> None:
        replay = _pasted("hello ")

        state = replay.apply_paste("world")

        assert state.spans == (
            PasteSpan(0, "hello ", "hello "),
            PasteSpan(1, "world", "world"),
        )

    def test_should_attribute_text_typed_after_a_paste_to_that_paste(self) -> None:
        replay = _pasted("hello")

        state = _type(replay, "!")

        assert state.spans == (PasteSpan(0, "hello", "hello!"),)

    def test_should_attribute_text_typed_before_the_first_paste_to_it(self) -> None:
        replay = EditReplay()
        _type(replay, "so ")

        state = replay.apply_paste("hello")

        assert state.spans == (PasteSpan(0, "hello", "so hello"),)

    def test_should_attribute_a_correction_between_pastes_to_the_earlier_one(
        self,
    ) -> None:
        replay = _pasted("hello")
        replay.apply_paste("world")
        for _ in range(len("world")):
            replay.apply_key(KeyStroke(VK_LEFT, ""))

        state = _type(replay, " ")

        assert state.spans == (
            PasteSpan(0, "hello", "hello "),
            PasteSpan(1, "world", "world"),
        )

    def test_should_keep_a_paste_out_of_the_span_it_was_dropped_into(self) -> None:
        replay = _pasted("alpha beta")
        for _ in range(len(" beta")):
            replay.apply_key(KeyStroke(VK_LEFT, ""))

        state = replay.apply_paste("XX")

        assert state.spans == (
            PasteSpan(0, "alpha beta", "alpha"),
            PasteSpan(1, "XX", "XX"),
        )

    def test_should_report_a_deleted_paste_with_an_empty_final_text(self) -> None:
        replay = _pasted("abc")

        for _ in range(len("abc")):
            replay.apply_key(KeyStroke(VK_BACK, ""))
        state = replay.snapshot()

        assert state.text == ""
        assert state.spans == (PasteSpan(0, "abc", ""),)
        assert state.pasted_text == "abc"

    def test_should_report_the_surviving_paste_when_an_earlier_one_is_deleted(
        self,
    ) -> None:
        replay = _pasted("gone")
        replay.apply_paste("kept")
        replay.apply_key(KeyStroke(VK_HOME, "", ctrl=True))
        for _ in range(4):
            replay.apply_key(KeyStroke(VK_DELETE, ""))

        state = replay.snapshot()

        assert state.spans == (
            PasteSpan(0, "gone", ""),
            PasteSpan(1, "kept", "kept"),
        )


class TestFlags:
    def test_should_stay_reliable_for_plain_typing(self) -> None:
        replay = _pasted("hello")

        state = _type(replay, " there")

        assert state.reliable is True

    def test_should_become_unreliable_after_a_mouse_click(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(ord("A"), "a", click_since_last=True))

        assert EditFlag.MOUSE_CLICK in state.flags
        assert state.reliable is False

    def test_should_flag_undo_on_control_z(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_Z, "", ctrl=True))

        assert EditFlag.UNDO in state.flags
        assert state.text == "hello"
        assert state.reliable is False

    def test_should_flag_undo_on_alt_backspace(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_BACK, "", alt=True))

        assert EditFlag.UNDO in state.flags
        assert state.text == "hello"

    def test_should_flag_a_paste_we_did_not_make(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_V, "", ctrl=True))

        assert EditFlag.FOREIGN_PASTE in state.flags
        assert state.reliable is False

    def test_should_flag_an_ime_composition(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_PROCESSKEY, "x"))

        assert EditFlag.IME in state.flags
        assert state.text == "hello"
        assert state.reliable is False

    def test_should_flag_tab_without_changing_the_text(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_TAB, "\t"))

        assert state.flags == frozenset({EditFlag.TAB})
        assert state.text == "hello"

    def test_should_become_unreliable_after_tab_leaves_the_control(self) -> None:
        replay = _pasted("hello")
        replay.apply_key(KeyStroke(VK_TAB, "\t"))

        state = _type(replay, "X")

        assert state.reliable is False

    def test_should_flag_escape_without_changing_the_text(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_ESCAPE, ""))

        assert state.flags == frozenset({EditFlag.ESCAPE})
        assert state.text == "hello"

    def test_should_become_unreliable_after_escape_clears_the_box(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_ESCAPE, ""))

        assert state.reliable is False

    def test_should_commit_a_vertical_arrow_edit_as_unreliable(self) -> None:
        replay = _pasted("line one\nline two")
        replay.apply_key(KeyStroke(VK_UP, ""))
        _type(replay, "X")

        state = replay.apply_key(KeyStroke(VK_RETURN, ""))

        assert state.committed is True
        assert state.reliable is False

    def test_should_flag_a_paste_that_lands_inside_earlier_text(self) -> None:
        replay = _pasted("alpha beta")
        for _ in range(len(" beta")):
            replay.apply_key(KeyStroke(VK_LEFT, ""))

        state = replay.apply_paste("XX")

        assert state.text == "alphaXX beta"
        assert EditFlag.SPLIT_PASTE in state.flags
        assert state.reliable is False

    def test_should_not_flag_a_paste_appended_after_earlier_text(self) -> None:
        replay = _pasted("alpha ")

        state = replay.apply_paste("beta")

        assert EditFlag.SPLIT_PASTE not in state.flags
        assert state.reliable is True

    def test_should_not_flag_a_paste_that_replaces_the_whole_box(self) -> None:
        replay = _pasted("alpha")
        replay.apply_key(KeyStroke(VK_A, "", ctrl=True))

        state = replay.apply_paste("beta")

        assert state.text == "beta"
        assert EditFlag.SPLIT_PASTE not in state.flags

    def test_should_flag_control_home_as_leaving_the_span(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_HOME, "", ctrl=True))

        assert state.flags == frozenset({EditFlag.CARET_BEFORE_SPAN})
        assert state.reliable is True

    def test_should_flag_control_end_as_leaving_the_span(self) -> None:
        replay = _pasted("hello")

        state = replay.apply_key(KeyStroke(VK_END, "", ctrl=True))

        assert state.flags == frozenset({EditFlag.CARET_AFTER_SPAN})

    def test_should_keep_flags_once_they_are_set(self) -> None:
        replay = _pasted("hello")
        replay.apply_key(KeyStroke(VK_Z, "", ctrl=True))

        state = _type(replay, "!")

        assert EditFlag.UNDO in state.flags


class TestReturnModifiers:
    @pytest.mark.parametrize("vk", [VK_UP, VK_DOWN, VK_PRIOR, VK_NEXT, VK_TAB, VK_ESCAPE])
    def test_should_flag_navigation_with_control_held(self, vk: int) -> None:
        replay = _pasted("line one\nline two")

        state = replay.apply_key(KeyStroke(vk, "", ctrl=True))

        assert state.reliable is False

    def test_should_commit_on_plain_return(self):
        replay = EditReplay()
        replay.apply_paste("hello")

        state = replay.apply_key(KeyStroke(vk=VK_RETURN, char=""))

        assert state.committed is True
        assert state.text == "hello"

    def test_should_not_commit_on_shift_return(self):
        replay = EditReplay()
        replay.apply_paste("hello")

        state = replay.apply_key(KeyStroke(vk=VK_RETURN, char="", shift=True))

        assert state.committed is False
        assert state.text == "hello\n"

    def test_should_commit_on_control_return(self):
        replay = EditReplay()
        replay.apply_paste("hello")

        state = replay.apply_key(KeyStroke(vk=VK_RETURN, char="", ctrl=True))

        assert state.committed is True
        assert state.text == "hello"

    def test_should_keep_typing_after_a_shift_return(self):
        replay = EditReplay()
        replay.apply_paste("hi")
        replay.apply_key(KeyStroke(vk=VK_RETURN, char="", shift=True))

        state = replay.apply_key(KeyStroke(vk=ord("A"), char="a"))

        assert state.text == "hi\na"
