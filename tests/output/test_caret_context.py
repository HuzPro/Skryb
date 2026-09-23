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


class TestEdgesOfTheText:
    @pytest.mark.parametrize(
        ("before", "selected", "after", "text", "expected"),
        [
            ("Hello there.", "", "", "How are you?", " How are you?"),
            ("Hello there.", "", "", "how are you?", " How are you?"),
            ("Hello there?", "", "", "Sure.", " Sure."),
            ("Hello there!", "", "", "sure.", " Sure."),
            ("Hello there", "", "", "How are you?", " how are you?"),
            ("Hello there,", "", "", "How are you?", " how are you?"),
            ("Buy these:", "", "", "Apples.", " apples."),
            ("Hello there. ", "", "", "how are you?", "How are you?"),
            ("Hello there.\n", "", "", "how are you?", "How are you?"),
            ('He said "hi."', "", "", "then we left.", ' Then we left.'),
            ('the word "hi"', "", "", "Is odd.", " is odd."),
            ("(see above)", "", "", "Then we left.", " then we left."),
            ("(see above.)", "", "", "then we left.", " Then we left."),
            ('He said "', "", "", "Hello.", "Hello."),
            ("Wait…", "", "", "Maybe.", " Maybe."),
        ],
        ids=[
            "after a full stop",
            "after a full stop, lowercase dictation",
            "after a question mark",
            "after an exclamation mark",
            "after a bare word",
            "after a comma",
            "after a colon",
            "after a full stop and a space",
            "after a newline",
            "after a quoted sentence",
            "after a quoted word",
            "after a closing bracket",
            "after a bracketed sentence",
            "after an opening quote",
            "after an ellipsis",
        ],
    )
    def test_should_fit_an_append_at_the_end(
        self, before: str, selected: str, after: str, text: str, expected: str,
    ) -> None:
        assert fit_to_context(text, CaretContext(before, selected, after)) == expected

    @pytest.mark.parametrize(
        ("before", "selected", "after", "text", "expected"),
        [
            ("", "", "Hello there.", "Good morning.", "Good morning. "),
            ("", "", "Hello there.", "good morning", "Good morning "),
            ("", "", "hello there.", "Well.", "Well "),
            ("", "", "\nHello there.", "Good morning.", "Good morning."),
            ("", "", " Hello there.", "good morning.", "Good morning."),
        ],
        ids=[
            "before a sentence",
            "before a sentence, unpunctuated dictation",
            "before a lowercase word",
            "before a newline",
            "before a space",
        ],
    )
    def test_should_fit_a_prepend_at_the_start(
        self, before: str, selected: str, after: str, text: str, expected: str,
    ) -> None:
        assert fit_to_context(text, CaretContext(before, selected, after)) == expected

    @pytest.mark.parametrize(
        ("before", "selected", "after", "text", "expected"),
        [
            ("Can you do this", ".", "", "?", "?"),
            ("Can you do this", ".", "", "Right?", " right?"),
            ("Can you do this", ".", "", "Is it done?", " is it done?"),
            ("Can you do this", ".", "", ", right?", ", right?"),
            ("Can you do this", " .", "", "?", "?"),
            ("Can you do this", ".", " Thanks.", "?", "?"),
            ("Can you do this", ".", " thanks", "?", "?"),
            ("One. Two", ".", " Three.", "Four.", " four."),
        ],
        ids=[
            "with a bare question mark",
            "with a short question",
            "with a longer question",
            "with a punctuation-led continuation",
            "with its leading space selected too",
            "before another sentence",
            "before a lowercase word",
            "with words before another sentence",
        ],
    )
    def test_should_fit_a_replaced_trailing_full_stop(
        self, before: str, selected: str, after: str, text: str, expected: str,
    ) -> None:
        assert fit_to_context(text, CaretContext(before, selected, after)) == expected

    @pytest.mark.parametrize(
        ("before", "selected", "after", "text", "expected"),
        [
            ("Hello there", " friend.", "", "Pal.", " pal."),
            ("", "Hello ", "there.", "Hi.", "Hi "),
            ("One. ", "Two. Three", " four.", "Five six.", "Five six"),
            ("One.", " Two.", "", "three.", " Three."),
            ("", "Hello.", "", "hi.", "Hi."),
        ],
        ids=[
            "last word with its leading space",
            "first word with its trailing space",
            "across a sentence boundary",
            "last sentence with its leading space",
            "the whole text",
        ],
    )
    def test_should_fit_a_replaced_selection(
        self, before: str, selected: str, after: str, text: str, expected: str,
    ) -> None:
        assert fit_to_context(text, CaretContext(before, selected, after)) == expected
