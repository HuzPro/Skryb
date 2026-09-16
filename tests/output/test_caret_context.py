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
