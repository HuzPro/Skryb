import pytest

from voicetotext.output.caret_context import (
    NO_CONTEXT,
    CaretContext,
    fit_to_context,
)


class TestUnknownContext:
    @pytest.mark.parametrize("text", ["Hello there. ", "hello", "", "  spaced  "])
    def test_should_return_the_transcript_unchanged(self, text: str) -> None:
        assert fit_to_context(text, NO_CONTEXT) == text


class TestCapitalisationFromWhatPrecedes:
    def test_should_lowercase_a_sentence_initial_capital_mid_sentence(self) -> None:
        context = CaretContext("Hello, I'm ", "here for water", ". ")

        result = fit_to_context("Near harry potter. ", context)

        assert result == "near harry potter"

    def test_should_capitalise_after_a_sentence_ending(self) -> None:
        context = CaretContext("Some text. ", "", "")

        result = fit_to_context("hello there. ", context)

        assert result == "Hello there. "

    def test_should_capitalise_after_a_newline(self) -> None:
        context = CaretContext("Note:\n", "", "")

        result = fit_to_context("hello", context)

        assert result == "Hello"

    def test_should_keep_the_pronoun_i_capitalised(self) -> None:
        context = CaretContext("Hello, I'm ", "here for water", ". ")

        result = fit_to_context("I'm near harry potter.", context)

        assert result == "I'm near harry potter"

    def test_should_keep_an_acronym_capitalised(self) -> None:
        context = CaretContext("NASA said ", "", "")

        result = fit_to_context("NASA rocks.", context)

        assert result == "NASA rocks."

    def test_should_lowercase_after_a_comma(self) -> None:
        context = CaretContext("Hello, ", "", "")

        result = fit_to_context("Then we left.", context)

        assert result == "then we left."

    def test_should_lowercase_after_a_closing_bracket(self) -> None:
        context = CaretContext("(as agreed) ", "", "")

        result = fit_to_context("We left.", context)

        assert result == "we left."

    def test_should_lowercase_only_the_first_word(self) -> None:
        context = CaretContext("Hello, I'm ", "here for water", ". ")

        result = fit_to_context("Near Harry Potter.", context)

        assert result == "near Harry Potter"


class TestEdgeWhitespace:
    def test_should_restore_the_leading_space_of_the_selection(self) -> None:
        context = CaretContext("Hello, I'm", " here for water", ". ")

        result = fit_to_context("Near harry potter.", context)

        assert result == " near harry potter"

    def test_should_prefix_a_space_when_the_caret_touches_the_previous_word(
        self,
    ) -> None:
        context = CaretContext("one", "", "")

        result = fit_to_context("Two.", context)

        assert result == " two."

    def test_should_append_a_space_when_the_caret_touches_the_next_word(self) -> None:
        context = CaretContext("Hello, I'm ", "here", "water. ")

        result = fit_to_context("Near.", context)

        assert result == "near "

    def test_should_not_append_a_space_when_the_next_word_is_already_spaced(
        self,
    ) -> None:
        context = CaretContext("Hello, I'm ", "here", " water. ")

        result = fit_to_context("Near.", context)

        assert result == "near"


class TestTerminalPunctuationFromWhatFollows:
    def test_should_drop_punctuation_when_punctuation_follows(self) -> None:
        context = CaretContext("Hello, I'm ", "here for water", ". ")

        result = fit_to_context("Near harry potter. ", context)

        assert result == "near harry potter"

    def test_should_keep_punctuation_when_a_new_sentence_follows(self) -> None:
        context = CaretContext("one ", "two", " Three.")

        result = fit_to_context("Two.", context)

        assert result == "two."


class TestInvisibleCharacters:
    def test_should_capitalise_in_an_empty_composer_paragraph(self) -> None:
        context = CaretContext("\ufeff", "", "\ufeff\n")

        result = fit_to_context("Hello there.", context)

        assert result == "Hello there."

    def test_should_capitalise_at_the_start_of_an_empty_field(self) -> None:
        context = CaretContext("", "", "\n")

        result = fit_to_context("Hello there.", context)

        assert result == "Hello there."

    def test_should_capitalise_after_an_object_replacement_on_its_own_line(
        self,
    ) -> None:
        context = CaretContext("\ufffc\n\ufffc", "", "")

        result = fit_to_context("hello there.", context)

        assert result == "Hello there."

    def test_should_keep_the_full_stop_before_an_object_replacement(self) -> None:
        context = CaretContext("", "", "\n\ufffc\nSend")

        result = fit_to_context("Hello there.", context)

        assert result == "Hello there."


class TestUnreadableNeighbours:
    @pytest.mark.parametrize("text", ["Hello there.", "hello there."])
    def test_should_leave_the_case_alone_after_an_unknown_symbol(
        self, text: str,
    ) -> None:
        context = CaretContext("… ", "", "")

        assert fit_to_context(text, context) == text

    def test_should_keep_the_full_stop_before_an_unknown_symbol(self) -> None:
        context = CaretContext("", "", "*")

        result = fit_to_context("Hello there.", context)

        assert result == "Hello there."
