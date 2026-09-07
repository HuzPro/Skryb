import pytest

from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryOptions,
    LiteralRule,
    RegexRule,
)
from voicetotext.output.text_pipeline import (
    PipelineMode,
    TextPipeline,
    TokenKind,
    build_pipeline,
    tokenize,
)


def _batch(
    text: str,
    rules: tuple[LiteralRule | RegexRule, ...] = (),
    options: DictionaryOptions | None = None,
) -> str:
    return TextPipeline.for_batch(
        DictionaryConfig(options=options or DictionaryOptions(), rules=rules)
    ).process(text)


class TestTokenizer:
    def test_should_preserve_word_whitespace_and_punctuation_spans(self) -> None:
        tokens = tokenize("Hi,  café!")

        assert [(token.text, token.start, token.end, token.kind) for token in tokens] == [
            ("Hi", 0, 2, TokenKind.WORD),
            (",", 2, 3, TokenKind.PUNCTUATION),
            ("  ", 3, 5, TokenKind.WHITESPACE),
            ("café", 5, 9, TokenKind.WORD),
            ("!", 9, 10, TokenKind.PUNCTUATION),
        ]


class TestFillerRemoval:
    @pytest.mark.parametrize("filler", ["um", "uh", "uhh", "umm", "erm"])
    def test_should_remove_only_the_exact_fillers(self, filler: str) -> None:
        options = DictionaryOptions(remove_fillers=True)

        result = _batch(f"I {filler} agree", options=options)

        assert result == "I agree"

    @pytest.mark.parametrize("filler", ["ummmm", "uhhh", "ermmm"])
    def test_should_remove_elongated_fillers(self, filler: str) -> None:
        options = DictionaryOptions(remove_fillers=True)

        result = _batch(f"I {filler} agree", options=options)

        assert result == "I agree"

    def test_should_match_fillers_case_insensitively(self) -> None:
        options = DictionaryOptions(remove_fillers=True)

        assert _batch("UHH yes", options=options) == "yes"

    def test_should_not_remove_fillers_inside_words_or_expand_the_list(self) -> None:
        options = DictionaryOptions(remove_fillers=True)

        result = _batch("a drum and an umbrella like so well actually", options=options)

        assert result == "a drum and an umbrella like so well actually"

    def test_should_leave_one_comma_when_a_filler_is_comma_wrapped(self) -> None:
        options = DictionaryOptions(remove_fillers=True)

        result = _batch("I was, um, thinking", options=options)

        assert result == "I was, thinking"

    def test_should_drop_the_trailing_comma_of_a_leading_filler(self) -> None:
        options = DictionaryOptions(remove_fillers=True)

        result = _batch("Um, hello", options=options)

        assert result == "hello"

    def test_should_drop_the_leading_comma_of_a_sentence_final_filler(self) -> None:
        options = DictionaryOptions(remove_fillers=True)

        result = _batch("I was, um.", options=options)

        assert result == "I was. "

    def test_should_preserve_text_when_the_module_is_off(self) -> None:
        assert _batch("I um agree") == "I um agree"

    def test_should_preserve_unrelated_existing_spacing(self) -> None:
        options = DictionaryOptions(remove_fillers=True)

        assert _batch("I  agree um now", options=options) == "I  agree now"


class TestSpokenCommands:
    @pytest.mark.parametrize(("spoken", "expected"), [
        ("comma", "hello, world"),
        ("period", "hello. World"),
        ("question mark", "hello? World"),
        ("exclamation mark", "hello! World"),
        ("colon", "hello: world"),
        ("semicolon", "hello; world"),
        ("dash", "hello- world"),
    ])
    def test_should_insert_attached_punctuation(
        self, spoken: str, expected: str,
    ) -> None:
        options = DictionaryOptions(spoken_commands=True)

        result = _batch(f"hello {spoken} world", options=options)

        assert result == expected

    @pytest.mark.parametrize(("spoken", "expected"), [
        ("new line", "first\nsecond"),
        ("new paragraph", "first\n\nSecond"),
        ("tab", "first\tsecond"),
    ])
    def test_should_trim_spaces_around_layout_commands(
        self, spoken: str, expected: str,
    ) -> None:
        options = DictionaryOptions(spoken_commands=True)

        result = _batch(f"first {spoken} second", options=options)

        assert result == expected

    def test_should_reverse_spacing_for_openers_and_attach_closers(self) -> None:
        options = DictionaryOptions(spoken_commands=True)

        result = _batch(
            "say open quote hello close quote and open paren yes close paren",
            options=options,
        )

        assert result == 'say "hello" and (yes)'

    def test_should_recognize_commands_case_insensitively(self) -> None:
        options = DictionaryOptions(spoken_commands=True)

        assert _batch("hello PERIOD next", options=options) == "hello. Next"

    def test_should_escape_a_single_command_word(self) -> None:
        options = DictionaryOptions(spoken_commands=True)

        result = _batch("say literal comma then comma", options=options)

        assert result == "say comma then,"

    def test_should_escape_an_entire_multi_word_command(self) -> None:
        options = DictionaryOptions(spoken_commands=True)

        result = _batch("literal question mark then period", options=options)

        assert result == "question mark then. "

    def test_should_keep_literal_and_the_next_word_when_no_command_follows(
        self,
    ) -> None:
        options = DictionaryOptions(spoken_commands=True)

        result = _batch("a literal interpretation", options=options)

        assert result == "a literal interpretation"

    def test_should_still_apply_later_commands_after_an_unused_literal(self) -> None:
        options = DictionaryOptions(spoken_commands=True)

        result = _batch("a literal interpretation comma really", options=options)

        assert result == "a literal interpretation, really"

    def test_should_capitalize_after_an_opener_following_a_terminal(self) -> None:
        options = DictionaryOptions(spoken_commands=True)

        result = _batch(
            "hello period open quote next close quote", options=options,
        )

        assert result == 'hello. "Next"'

    def test_should_preserve_commands_when_the_module_is_off(self) -> None:
        assert _batch("hello comma world") == "hello comma world"


class TestTrailingSentenceSpace:
    @pytest.mark.parametrize("terminator", [".", "?", "!"])
    def test_should_append_a_space_after_a_sentence_terminator(
        self, terminator: str,
    ) -> None:
        assert _batch(f"testing 1 2 3{terminator}") == f"testing 1 2 3{terminator} "

    def test_should_separate_back_to_back_dictations(self) -> None:
        first = _batch("testing one two three.")
        second = _batch("and then I stopped.")

        assert first + second == "testing one two three. and then I stopped. "

    def test_should_not_append_a_space_without_a_terminator(self) -> None:
        assert _batch("no terminator here") == "no terminator here"

    def test_should_not_double_an_existing_trailing_space(self) -> None:
        assert _batch("already spaced. ") == "already spaced. "

    def test_should_leave_a_comma_ended_phrase_alone(self) -> None:
        assert _batch("a clause,") == "a clause,"

    def test_should_leave_empty_text_alone(self) -> None:
        assert _batch("") == ""


class TestRuleStage:
    def test_should_apply_the_longest_multi_alias_literal_match(self) -> None:
        rules = (
            LiteralRule(("git",), "Git"),
            LiteralRule(("git hub", "get hub"), "GitHub"),
        )

        result = _batch("git hub and get hub use git", rules)

        assert result == "GitHub and GitHub use Git"

    def test_should_match_case_insensitively_and_emit_verbatim(self) -> None:
        rules = (LiteralRule(("git hub",), "GitHub"),)

        assert _batch("GIT HUB", rules) == "GitHub"

    def test_should_respect_case_sensitive_literal_rules(self) -> None:
        rules = (LiteralRule(("god",), "God", case_sensitive=True),)

        assert _batch("god GOD", rules) == "God GOD"

    def test_should_allow_unbounded_literal_rules(self) -> None:
        rules = (LiteralRule(("cat",), "dog", whole_words=False),)

        assert _batch("concatenate cat", rules) == "condogenate dog"

    def test_should_glue_aliases_with_edge_spaces_to_their_neighbours(self) -> None:
        rules = (LiteralRule((" underscore ",), "_"),)

        assert _batch("hello underscore world", rules) == "hello_world"
        assert _batch("underscore world", rules) == "_world"
        assert _batch("hello underscore", rules) == "hello_"

    def test_should_glue_only_across_horizontal_whitespace(self) -> None:
        rules = (LiteralRule((" underscore ",), "_"),)

        assert _batch("foo\n\nunderscore bar", rules) == "foo\n\n_bar"

    def test_should_glue_only_the_side_the_alias_marks(self) -> None:
        assert (
            _batch("hello underscore world", (LiteralRule(("underscore ",), "_"),))
            == "hello _world"
        )
        assert (
            _batch("hello underscore world", (LiteralRule((" underscore",), "_"),))
            == "hello_ world"
        )

    def test_should_glue_across_tabs(self) -> None:
        rules = (LiteralRule((" underscore ",), "_"),)

        assert _batch("hello \t underscore \t world", rules) == "hello_world"

    def test_should_keep_whole_word_matching_for_glued_aliases(self) -> None:
        rules = (LiteralRule((" underscore ",), "_"),)

        assert _batch("hello underscored world", rules) == "hello underscored world"

    def test_should_glue_mid_word_when_whole_words_is_off(self) -> None:
        rules = (LiteralRule((" underscore ",), "_", whole_words=False),)

        assert _batch("hello underscored world", rules) == "hello_d world"

    def test_should_apply_regex_capture_replacements(self) -> None:
        rules = (RegexRule(r"(\d{2})-(\d{2})", r"\2/\1"),)

        assert _batch("Date 12-31", rules) == "Date 31/12"

    def test_should_skip_disabled_rules(self) -> None:
        rules = (
            LiteralRule(("one",), "two", enabled=False),
            RegexRule("three", "four", enabled=False),
        )

        assert _batch("one three", rules) == "one three"

    def test_should_apply_literal_and_regex_rules_in_one_non_cascading_pass(
        self,
    ) -> None:
        rules = (
            LiteralRule(("foo",), "bar"),
            RegexRule("bar", "baz"),
        )

        assert _batch("foo bar", rules) == "bar baz"


class TestPhoneticRules:
    def test_should_correct_a_phonetic_multi_word_candidate(
        self, app_caplog: pytest.LogCaptureFixture,
    ) -> None:
        rules = (LiteralRule(("github",), "GitHub", phonetic=True),)

        result = _batch("open get hub now", rules)

        assert result == "open GitHub now"
        assert "phonetic: 'get hub' -> 'GitHub'" in app_caplog.messages

    def test_should_use_alias_and_replacement_metaphone_keys(self) -> None:
        rules = (LiteralRule(("repository",), "GitHub", phonetic=True),)

        assert _batch("get hub", rules) == "GitHub"

    def test_should_not_change_an_existing_replacement(self) -> None:
        rules = (LiteralRule(("get hub",), "GitHub", phonetic=True),)

        assert _batch("github", rules) == "github"

    def test_should_not_phonetically_replace_single_function_words(self) -> None:
        rules = (LiteralRule(("too",), "Two", phonetic=True),)

        assert _batch("to", rules) == "to"

    def test_should_let_an_exact_match_win_any_overlap(self) -> None:
        rules = (
            LiteralRule(("gethub",), "GitHub", phonetic=True),
            LiteralRule(("hub",), "HUB"),
        )

        assert _batch("get hub", rules) == "get HUB"

    def test_should_choose_the_strictly_closer_rule_for_the_same_span(self) -> None:
        rules = (
            LiteralRule(("github",), "GitHub", phonetic=True),
            LiteralRule(("gathub",), "Other", phonetic=True),
        )

        assert _batch("get hub", rules) == "GitHub"

    def test_should_skip_a_span_when_different_rules_tie(self) -> None:
        rules = (
            LiteralRule(("github",), "First", phonetic=True),
            LiteralRule(("github",), "Second", phonetic=True),
        )

        assert _batch("get hub", rules) == "get hub"

    def test_should_limit_candidates_to_four_words(self) -> None:
        rules = (LiteralRule(("one two three four five",), "Matched", phonetic=True),)

        assert _batch("won two three four five", rules) == "won two three four five"


class TestNumbers:
    def test_should_run_alpha2digit_for_english(self) -> None:
        options = DictionaryOptions(numbers=True)

        assert _batch("I need twenty one items", options=options) == "I need 21 items"

    @pytest.mark.parametrize(("spoken", "expected"), [
        ("twenty percent", "20%"),
        ("five dollars and six cents", "$5.06"),
        ("five dollars", "$5"),
        ("twelve point five dollars", "$12.5"),
    ])
    def test_should_apply_only_the_supported_wrappers(
        self, spoken: str, expected: str,
    ) -> None:
        assert _batch(spoken, options=DictionaryOptions(numbers=True)) == expected

    def test_should_apply_number_formatting_after_dictionary_rules(self) -> None:
        rules = (LiteralRule(("a score",), "twenty"),)

        assert _batch(
            "a score percent", rules, DictionaryOptions(numbers=True),
        ) == "20%"

    def test_should_leave_number_words_when_the_module_is_off(self) -> None:
        assert _batch("twenty percent") == "twenty percent"


class TestStreamingPipeline:
    def test_should_keep_right_glue_between_writer_updates(self) -> None:
        from tests.fakes import FakeTextOutputHandler
        from voicetotext.output.text_filter_decorator import TextFilterDecorator
        from voicetotext.transcription.streaming.streaming_output_writer import (
            StreamingOutputWriter,
        )

        config = DictionaryConfig(rules=(LiteralRule((" underscore ",), "_"),))
        output = FakeTextOutputHandler()
        writer = StreamingOutputWriter(TextFilterDecorator(
            output, TextPipeline.for_streaming(config),
        ))

        for text in ("foo", "foo underscore", "foo underscore bar"):
            writer.emit(text)

        assert "".join(output.outputs) == "foo_bar"
        writer.emit("foo underscore bar underscore")
        writer.reset()
        writer.emit("new")
        writer.emit("new dictation")
        assert "".join(output.outputs) == "foo_bar_new dictation"

    @pytest.mark.parametrize("spoken", ["underscore underscore", "underscore dash"])
    def test_should_replace_adjacent_glued_aliases(self, spoken: str) -> None:
        config = DictionaryConfig(rules=(
            LiteralRule((" underscore ",), "_"),
            LiteralRule((" dash ",), "-"),
        ))

        expected = "__" if spoken.endswith("underscore") else "_-"
        for mode in (PipelineMode.BATCH, PipelineMode.STREAMING):
            assert build_pipeline(config, mode).process(spoken) == expected

    def test_should_keep_fillers_single_token_literals_and_regex(self) -> None:
        config = DictionaryConfig(
            options=DictionaryOptions(remove_fillers=True, spoken_commands=True),
            rules=(
                LiteralRule(("git", "git hub"), "Git"),
                RegexRule(r"\bissue (\d+)\b", r"#\1"),
            ),
        )

        result = build_pipeline(config, PipelineMode.STREAMING).process(
            "um git hub issue 42 comma"
        )

        assert result == "Git hub #42 comma"

    def test_should_keep_alias_glue_in_streaming(self) -> None:
        config = DictionaryConfig(rules=(LiteralRule((" underscore ",), "_"),))

        result = build_pipeline(config, PipelineMode.STREAMING).process(
            " underscore"
        )

        assert result == "_"

    def test_should_keep_phrase_rules_and_commands_in_batch(self) -> None:
        config = DictionaryConfig(
            options=DictionaryOptions(spoken_commands=True),
            rules=(LiteralRule(("git hub",), "GitHub"),),
        )

        result = build_pipeline(config, PipelineMode.BATCH).process(
            "git hub comma yes"
        )

        assert result == "GitHub, yes"

    def test_should_exclude_numbers_and_phonetic_rules(self) -> None:
        config = DictionaryConfig(
            options=DictionaryOptions(numbers=True, punctuation_model=True),
            rules=(LiteralRule(("github",), "GitHub", phonetic=True),),
        )

        result = TextPipeline.for_streaming(config).process("get hub twenty percent")

        assert result == "get hub twenty percent"


def _everything_on() -> DictionaryConfig:
    return DictionaryConfig(
        options=DictionaryOptions(
            remove_fillers=True, spoken_commands=True, numbers=True,
        ),
        rules=(LiteralRule(spoken=("cloud",), replacement="Claude"),),
    )


class TestVerbatimPipeline:
    def test_should_return_the_text_untouched(self) -> None:
        pipeline = build_pipeline(_everything_on(), PipelineMode.VERBATIM)
        spoken = "um cloud said uh twenty five percent period"

        assert pipeline.process(spoken) == spoken

    def test_should_keep_what_the_batch_pipeline_would_rewrite(self) -> None:
        config = _everything_on()
        spoken = "um cloud said uh twenty five percent period"

        verbatim = build_pipeline(config, PipelineMode.VERBATIM).process(spoken)
        batch = build_pipeline(config, PipelineMode.BATCH).process(spoken)

        assert "um" in verbatim and "uh" in verbatim
        assert "cloud" in verbatim
        assert batch != verbatim
