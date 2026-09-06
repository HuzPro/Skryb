import tkinter
from collections.abc import Iterator

import pytest

from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryOptions,
    LiteralRule,
    RegexRule,
)
from voicetotext.settings.tk_pages.dictionary_page import TkDictionaryPage


@pytest.fixture
def page(tk_root: tkinter.Misc) -> Iterator[TkDictionaryPage]:
    widget = TkDictionaryPage(tk_root)
    yield widget
    widget.destroy()


def _type_rule(
    page: TkDictionaryPage, spoken: str, replacement: str,
) -> None:
    page.spoken_entry().delete(0, "end")
    page.spoken_entry().insert(0, spoken)
    page.replacement_entry().delete(0, "end")
    page.replacement_entry().insert(0, replacement)


class TestInitialDictionary:
    def test_should_list_literal_and_regex_rules(self, page: TkDictionaryPage) -> None:
        page.set_dictionary(DictionaryConfig(rules=(
            LiteralRule(("git hub", "get hub"), "GitHub"),
            RegexRule(r"\bum+\b", ""),
        )))

        assert page.rule_labels() == [
            "[literal] git hub, get hub  ->  GitHub",
            r"[regex] \bum+\b  ->  ",
        ]

    def test_should_start_clean(self, page: TkDictionaryPage) -> None:
        page.set_dictionary(
            DictionaryConfig(rules=(LiteralRule(("god",), "God"),))
        )

        assert page.is_dirty() is False


class TestOptions:
    def test_should_stage_filler_and_command_toggles(
        self, page: TkDictionaryPage,
    ) -> None:
        page.set_dictionary(DictionaryConfig())

        page.fillers_check().invoke()
        page.commands_check().invoke()

        assert page.dictionary().options == DictionaryOptions(
            remove_fillers=True, spoken_commands=True,
        )
        assert page.is_dirty() is True

    def test_should_preserve_phase_b_options(self, page: TkDictionaryPage) -> None:
        page.set_dictionary(DictionaryConfig(options=DictionaryOptions(
            numbers=True, punctuation_model=True,
        )))

        page.fillers_check().invoke()

        assert page.dictionary().options == DictionaryOptions(
            remove_fillers=True, numbers=True, punctuation_model=True,
        )

    def test_should_stage_number_and_punctuation_toggles(
        self, page: TkDictionaryPage,
    ) -> None:
        page.set_dictionary(DictionaryConfig())

        page.numbers_check().invoke()
        page.punctuation_check().invoke()

        assert page.dictionary().options == DictionaryOptions(
            numbers=True, punctuation_model=True,
        )
        assert page.is_dirty() is True


class TestAdd:
    def test_should_stage_a_multi_alias_literal_rule(
        self, page: TkDictionaryPage,
    ) -> None:
        _type_rule(page, "git hub, get hub", "GitHub")
        page.case_check().invoke()

        page.add_button().invoke()

        assert page.dictionary().rules == (
            LiteralRule(
                ("git hub", "get hub"), "GitHub", case_sensitive=True,
            ),
        )

    def test_should_keep_edge_spaces_as_glue_on_every_alias(
        self, page: TkDictionaryPage,
    ) -> None:
        _type_rule(page, " underscore, under score ", "_")

        page.add_button().invoke()

        assert page.dictionary().rules == (
            LiteralRule((" underscore ", " under score "), "_"),
        )
        assert page.rule_labels() == ['[literal] " underscore, under score "  ->  _']

    def test_should_treat_a_glued_alias_as_a_duplicate_of_the_plain_one(
        self, page: TkDictionaryPage,
    ) -> None:
        page.set_dictionary(
            DictionaryConfig(rules=(LiteralRule(("underscore",), "_"),))
        )
        _type_rule(page, " underscore ", "_")

        page.add_button().invoke()

        assert len(page.dictionary().rules) == 1
        assert "another" in page.message().lower()

    def test_should_stage_a_regex_rule_with_an_empty_replacement(
        self, page: TkDictionaryPage,
    ) -> None:
        page.regex_radio().invoke()
        _type_rule(page, r"\bum+\b", "")

        page.add_button().invoke()

        assert page.dictionary().rules == (RegexRule(r"\bum+\b", ""),)

    def test_should_stage_a_phonetic_literal_rule(
        self, page: TkDictionaryPage,
    ) -> None:
        _type_rule(page, "github", "GitHub")
        page.phonetic_check().invoke()

        page.add_button().invoke()

        assert page.dictionary().rules == (
            LiteralRule(("github",), "GitHub", phonetic=True),
        )

    def test_should_clear_the_entry_fields(self, page: TkDictionaryPage) -> None:
        _type_rule(page, "god", "God")

        page.add_button().invoke()

        assert page.spoken_entry().get() == ""
        assert page.replacement_entry().get() == ""


class TestValidation:
    def test_should_reject_empty_aliases(self, page: TkDictionaryPage) -> None:
        _type_rule(page, "", "God")

        page.add_button().invoke()

        assert page.dictionary().rules == ()
        assert "required" in page.message()
        assert page.is_dirty() is False

    def test_should_reject_an_invalid_regex(self, page: TkDictionaryPage) -> None:
        page.regex_radio().invoke()
        _type_rule(page, "[", "broken")

        page.add_button().invoke()

        assert page.dictionary().rules == ()
        assert "invalid" in page.message().lower()

    def test_should_reject_an_invalid_regex_replacement(
        self, page: TkDictionaryPage,
    ) -> None:
        page.regex_radio().invoke()
        _type_rule(page, "ok", r"\2")

        page.add_button().invoke()

        assert page.dictionary().rules == ()
        assert "invalid" in page.message().lower()

    def test_should_reject_an_alias_used_by_another_rule(
        self, page: TkDictionaryPage,
    ) -> None:
        page.set_dictionary(
            DictionaryConfig(rules=(LiteralRule(("git hub",), "GitHub"),))
        )
        _type_rule(page, "git hub, github", "GitHub")

        page.add_button().invoke()

        assert len(page.dictionary().rules) == 1
        assert "another" in page.message().lower()


class TestEdit:
    def test_should_replace_the_selected_rule_in_place(
        self, page: TkDictionaryPage,
    ) -> None:
        page.set_dictionary(DictionaryConfig(rules=(
            LiteralRule(("god",), "God"),
            RegexRule("python", "Python"),
        )))
        page.select_rule(0)
        _type_rule(page, "god, gosh", "Deity")

        page.update_button().invoke()

        assert page.dictionary().rules == (
            LiteralRule(("god", "gosh"), "Deity"),
            RegexRule("python", "Python"),
        )

    def test_should_populate_all_literal_editor_fields(
        self, page: TkDictionaryPage,
    ) -> None:
        page.set_dictionary(DictionaryConfig(rules=(
            LiteralRule(
                ("god", "gosh"), "God", case_sensitive=True, whole_words=False,
                phonetic=True,
            ),
        )))

        page.select_rule(0)

        assert page.spoken_entry().get() == "god, gosh"
        assert page.replacement_entry().get() == "God"
        assert page.case_check().instate(["selected"])
        assert not page.whole_words_check().instate(["selected"])
        assert page.phonetic_check().instate(["selected"])

    def test_should_show_glued_aliases_as_typed(
        self, page: TkDictionaryPage,
    ) -> None:
        page.set_dictionary(DictionaryConfig(rules=(
            LiteralRule((" underscore ", " under score "), "_"),
        )))

        page.select_rule(0)

        assert page.spoken_entry().get() == " underscore, under score "

    def test_should_disable_phonetic_for_regex_rules(
        self, page: TkDictionaryPage,
    ) -> None:
        page.regex_radio().invoke()

        assert page.phonetic_check().instate(["disabled"])

    def test_should_report_when_nothing_is_selected(
        self, page: TkDictionaryPage,
    ) -> None:
        _type_rule(page, "god", "Deity")

        page.update_button().invoke()

        assert "select a rule" in page.message().lower()


class TestDelete:
    def test_should_remove_the_selected_rule(self, page: TkDictionaryPage) -> None:
        page.set_dictionary(DictionaryConfig(rules=(
            LiteralRule(("god",), "God"),
            LiteralRule(("python",), "Python"),
        )))
        page.select_rule(0)

        page.delete_button().invoke()

        assert page.dictionary().rules == (LiteralRule(("python",), "Python"),)
        assert page.is_dirty() is True

    def test_should_report_when_nothing_is_selected(
        self, page: TkDictionaryPage,
    ) -> None:
        page.delete_button().invoke()

        assert "select a rule" in page.message().lower()
