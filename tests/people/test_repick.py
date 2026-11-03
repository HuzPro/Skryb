from dataclasses import replace
from pathlib import Path

import pytest

from skryb.output.caret_context import CaretContext
from skryb.output.dictionary_config import DictionaryConfig
from skryb.output.text_pipeline import TextPipeline
from skryb.people.repick import Repick, repick, suggest_heard_as
from skryb.people.terms import Suggestion, Term, load_people, update_people

_PEOPLE = (Term("Mikaela"), Term("Skryb", note="the app I'm building"), Term("Michaela"), Term("Odile", heard_as=("Odeel",)))


def _dictated(text: str, before: str, selected: str, after: str = "") -> tuple[str, list[Repick]]:
    picks: list[Repick] = []
    pipeline = TextPipeline.for_batch(
        DictionaryConfig(), caret_context=lambda: CaretContext(before, selected, after), terms=_PEOPLE, on_repick=picks.append,
    )
    return pipeline.process(text), picks


def test_re_dictating_a_name_over_itself_picks_the_other_person_with_that_name() -> None:
    picked = repick("Mikaela. ", "Mikaela ", _PEOPLE)
    assert picked == Repick("Michaela. ", ("Mikaela",), "Michaela")


def test_doing_it_again_cycles_back_to_the_first_spelling() -> None:
    picked = repick("Mikaela.", "Michaela", _PEOPLE)
    assert picked is not None and picked.text == "Mikaela." and picked.written == "Mikaela"


def test_an_entry_dictated_over_itself_becomes_the_ordinary_word_the_model_heard() -> None:
    picked = repick("Scribe.", "Skryb", _PEOPLE)
    assert picked is not None and picked.text == "scribe." and picked.written is None


def test_the_ordinary_word_dictated_over_itself_becomes_the_entry_and_is_suggested_as_heard_as() -> None:
    picked = repick("Scribe. ", "scribe ", _PEOPLE)
    assert picked == Repick("Skryb. ", ("Scribe",), "Skryb")


def test_two_people_with_one_name_cycle_between_them_whatever_third_spelling_the_model_wrote() -> None:
    first = repick("Mikayla.", "Michaela", _PEOPLE)
    second = repick("Mikayla.", "Mikayla", _PEOPLE)
    assert first is not None and first.text == "Mikaela."
    assert second is not None and second.text == "Mikaela."


def test_a_heard_as_spelling_in_the_selection_counts_as_its_entry() -> None:
    picked = repick("Odeel", "Odeel", _PEOPLE)
    assert picked is not None and picked.written == "Odile"


def test_a_possessive_keeps_its_s() -> None:
    picked = repick("Mikaela's.", "Mikaela's ", _PEOPLE)
    assert picked is not None and picked.text == "Michaela's."


@pytest.mark.parametrize(("text", "selected"), [
    ("Mikaela said hi.", "Mikaela "),
    ("Michaela's.", "Michaela "),
    ("the the", "the the"),
    ("Mikaela", "Mikaela and Michaela and Odile too"),
    ("Mikaela", ""),
    ("...", "Mikaela"),
    ("Skryb", "Skryb"),
    ("Rafferty.", "Rafferty "),
])
def test_dictating_something_else_or_over_nothing_known_is_left_alone(text: str, selected: str) -> None:
    assert repick(text, selected, _PEOPLE) is None


def test_with_no_list_nothing_is_re_picked() -> None:
    assert repick("Mikaela.", "Mikaela ", ()) is None


def test_the_re_pick_keeps_the_double_clicked_selections_trailing_space_and_mid_sentence_case() -> None:
    text, picks = _dictated("Mikaela.", "I sent it to ", "Mikaela ", "yesterday.")
    assert text == "Michaela "
    assert [pick.written for pick in picks] == ["Michaela"]


def test_the_entry_keeps_its_capital_when_it_replaces_the_ordinary_word_mid_sentence() -> None:
    text, _picks = _dictated("Scribe.", "I opened ", "scribe ", "again.")
    assert text == "Skryb "


def test_the_ordinary_word_takes_lower_case_mid_sentence_and_a_capital_at_a_sentence_start() -> None:
    assert _dictated("Scribe.", "I opened ", "Skryb ", "again.")[0] == "scribe "
    assert _dictated("Scribe.", "Done. ", "Skryb ", "is open.")[0] == "Scribe "


def test_a_dictation_with_no_selection_is_never_re_picked() -> None:
    text, picks = _dictated("Thanks.", "Done. ", "", "")
    assert text == "Thanks. "
    assert picks == []


def test_re_picks_become_suggestions_once_for_each_spelling_dictated_over(tmp_path: Path) -> None:
    path = tmp_path / "people.json"
    update_people(path, lambda people: replace(people, terms=_PEOPLE))

    suggest_heard_as(path, Repick("Skryb ", ("Scribe",), "Skryb"))
    suggest_heard_as(path, Repick("Skryb ", ("scribe",), "Skryb"))
    suggest_heard_as(path, Repick("Michaela ", ("Mikaela",), "Michaela"))
    suggest_heard_as(path, Repick("scribe ", ("Skryb",), None))

    assert load_people(path).suggestions == (Suggestion("scribe", "Skryb", 2),)
