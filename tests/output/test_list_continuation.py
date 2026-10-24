"""A dictation at the caret keeps the list layout of the text already there."""

import pytest

from skryb.output.caret_context import (
    CONTEXT_CHARS,
    NO_CONTEXT,
    CaretContext,
    ListLayout,
    fit_to_context,
    list_layout,
)


class TestListLayoutFollowsTheTextAtTheCaret:
    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [
            ("Testing 1, 2, 3", ".", ListLayout.INLINE),
            ("I'll meet you at the ", " tomorrow.", ListLayout.INLINE),
            ("Groceries: eggs, milk", "", ListLayout.INLINE),
            ("", "and the rest of the line", ListLayout.INLINE),
            ("1. Open the app\n2. Log in", "", ListLayout.VERTICAL),
            ("1. Open the app\n2. Log in", "\n3. Click settings", ListLayout.VERTICAL),
            ("1. Open the app\n2. Log in\n", "", ListLayout.VERTICAL),
            ("1. Open the app\n2. Log in\n3. ", "", ListLayout.VERTICAL),
            ("Packing:\n- tent\n- stove", "", ListLayout.VERTICAL),
            ("1) Café\n2) Zoë's place", "", ListLayout.VERTICAL),
            ("", "", None),
            ("First paragraph.\n", "", None),
            ("The steps are:", "", None),
            ("﻿", "\n", None),
        ],
        ids=[
            "inside-a-comma-list", "mid-sentence", "after-a-colon-list", "before-text-on-the-line",
            "end-of-a-numbered-item", "between-numbered-items", "empty-line-after-a-list",
            "empty-numbered-slot", "bulleted", "unicode-items", "empty-box", "fresh-line",
            "after-a-colon", "whatsapp-empty-composer",
        ],
    )
    def test_the_layout_matches_what_is_already_there(self, before, after, expected) -> None:
        assert list_layout(CaretContext(before, "", after)) == expected

    def test_a_caret_in_the_middle_of_an_item_keeps_the_text_on_that_line(self) -> None:
        context = CaretContext("1. Open the app\n2. Log", "", " in to the site")

        assert list_layout(context) is ListLayout.INLINE

    def test_a_marker_at_the_start_of_a_line_cut_off_by_the_reading_does_not_count(self) -> None:
        before = ("x" * CONTEXT_CHARS + " 2. Log in")[-CONTEXT_CHARS:]

        assert list_layout(CaretContext(before, "", "")) is ListLayout.INLINE

    def test_a_selection_being_replaced_leaves_the_choice_to_the_setting(self) -> None:
        assert list_layout(CaretContext("1. Open the app\n", "2. Log in", "")) is None

    def test_an_unreadable_caret_leaves_the_choice_to_the_setting(self) -> None:
        assert list_layout(NO_CONTEXT) is None


class TestVerticalListsGoOn:
    def test_items_after_the_end_of_an_item_start_a_new_line_with_the_next_number(self) -> None:
        context = CaretContext("1. Open the app\n2. Log in", "", "")

        assert fit_to_context("1. Click settings\n2. Save", context) == "\n3. Click settings\n4. Save"

    def test_a_single_spoken_item_becomes_the_next_item(self) -> None:
        context = CaretContext("1. Open the app\n2. Log in", "", "")

        assert fit_to_context("Click settings", context) == "\n3. Click settings"

    def test_an_empty_numbered_slot_is_filled_without_a_second_marker(self) -> None:
        context = CaretContext("1. Open the app\n2. Log in\n3. ", "", "")

        assert fit_to_context("1. Click settings\n2. Save", context) == "Click settings\n4. Save"

    def test_the_empty_line_after_a_list_gets_the_next_numbered_item(self) -> None:
        context = CaretContext("1. Open the app\n2. Log in\n", "", "")

        assert fit_to_context("Click settings", context) == "3. Click settings"

    def test_bullets_and_the_parenthesis_style_are_kept(self) -> None:
        bullets = CaretContext("Packing:\n- tent\n- stove", "", "")
        parens = CaretContext("1) Café\n2) Zoë's place", "", "")

        assert fit_to_context("1. Matches\n2. Torch", bullets) == "\n- matches\n- torch"
        assert fit_to_context("Crème brûlée", parens) == "\n3) Crème brûlée"

    def test_new_items_follow_the_case_of_the_item_before_them(self) -> None:
        capitalised = CaretContext("1. Open the app\n2. Log in\n3. ", "", "")
        lowercase = CaretContext("Packing:\n- tent\n- stove", "", "")

        assert fit_to_context("click settings", capitalised) == "Click settings"
        assert fit_to_context("Matches\nTorch\nI think a map", lowercase) == (
            "\n- matches\n- torch\n- I think a map"
        )

    def test_blank_lines_and_windows_line_breaks_in_the_answer_are_dropped(self) -> None:
        context = CaretContext("1. Open the app\r\n2. Log in", "", "")

        result = fit_to_context("\n1. Click settings\r\n\n2. Save\n", context)

        assert result == "\n3. Click settings\n4. Save"

    def test_numbering_carries_on_into_the_thousands(self) -> None:
        context = CaretContext("998. Item\n999. Item", "", "")
        spoken = "\n".join(f"{n}. Entry {n}" for n in range(1, 1001))

        lines = fit_to_context(spoken, context).split("\n")

        assert lines[0] == ""
        assert lines[1] == "1000. Entry 1"
        assert lines[-1] == "1999. Entry 1000"
        assert len(lines) == 1001


class TestNumberListsGoOnWithCommas:
    def test_spoken_numbers_after_a_number_list_join_it_before_its_full_stop(self) -> None:
        context = CaretContext("Testing 1, 2, 3", "", ".")

        result = fit_to_context("Four five six seven eight nine ten", context)

        assert context.before + result + context.after == "Testing 1, 2, 3, 4, 5, 6, 7, 8, 9, 10."

    def test_numbers_the_writing_model_already_separated_are_joined_the_same_way(self) -> None:
        context = CaretContext("Testing 1, 2, 3", "", "")

        assert fit_to_context("Four, five, six.", context) == ", 4, 5, 6."

    def test_a_single_small_number_is_still_written_as_a_digit(self) -> None:
        context = CaretContext("Seats 7, 8", "", "")

        assert fit_to_context("one", context) == ", 1"

    def test_a_long_run_into_the_thousands_is_joined_in_full(self) -> None:
        context = CaretContext("Rows 997, 998, 999", "", "")
        spoken = " ".join(str(n) for n in range(1000, 3000))

        assert fit_to_context(spoken, context) == ", " + ", ".join(str(n) for n in range(1000, 3000))

    @pytest.mark.parametrize(
        ("before", "spoken"),
        [
            ("Room 3", "four five"),
            ("Testing 1, 2, 3", "four apples"),
            ("Version 1.2, 3", "four"),
            ("Testing 1, 2, 3\nNext line", "four five"),
            ("Testing 1, 2, 3", ""),
        ],
        ids=[
            "one-number-is-not-a-list", "words-among-the-numbers", "a-decimal-before",
            "the-list-is-on-another-line", "blank-dictation",
        ],
    )
    def test_anything_else_is_left_to_the_usual_fitting(self, before, spoken) -> None:
        result = fit_to_context(spoken, CaretContext(before, "", ""))

        assert not result.startswith(",")
