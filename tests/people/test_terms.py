import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from skryb.output.dictionary_config import LiteralRule
from skryb.people.terms import (
    PeopleList,
    Suggestion,
    Term,
    accept_suggestion,
    dismiss_suggestion,
    import_vocabulary,
    load_people,
    people_rules,
    suggest,
    terms_from_json,
    update_people,
    with_terms,
)


def _saved(path: Path, *terms: Term) -> Path:
    update_people(path, lambda people: replace(people, terms=terms))
    return path


def test_the_list_comes_back_as_saved_with_notes_heard_as_spellings_and_unicode(tmp_path: Path) -> None:
    terms = (
        Term("Skryb", note="the app I'm building", heard_as=("Skribe", "Scryb")),
        Term("Ngozi Adébáyọ̀", phonetic=True),
        Term("東京オフィス", heard_as=("Tokyo office",)),
    )

    path = _saved(tmp_path / "people.json", *terms)

    assert load_people(path).terms == terms


def test_a_hand_edited_file_keeps_its_readable_entries_and_skips_the_broken_ones(tmp_path: Path) -> None:
    path = tmp_path / "people.json"
    path.write_text(json.dumps({"people": [
        {"written": "Mikaela", "heard_as": ["Mikayla", "  ", "mikaela", "Mikayla"]},
        {"written": "   "},
        {"note": "no name"},
        "Rafferty",
        {"written": "Odile", "phonetic": "yes"},
        {"written": "mikaela", "note": "the same name again"},
        {"written": "Michaela", "heard_as": ["Mikayla", "Mikaela", "Mihaela"]},
    ], "suggestions": [{"heard": "Mikala", "written": "Mikaela", "times": True}, {"heard": ""}],
        "dismissed": [["scribe", "Skryb"], ["only one"]]}), encoding="utf-8")

    people = load_people(path)

    assert people.terms == (Term("Mikaela", heard_as=("Mikayla",)), Term("Michaela", heard_as=("Mihaela",)))
    assert people.suggestions == (Suggestion("Mikala", "Mikaela", 1),)
    assert people.dismissed == (("scribe", "Skryb"),)


@pytest.mark.parametrize("content", [b"", b"[1, 2]", b"{not json", "Zürich".encode("utf-16")])
def test_a_file_that_cannot_be_read_is_an_empty_list(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "people.json"
    path.write_bytes(content)
    assert load_people(path) == PeopleList()


def test_no_file_is_an_empty_list(tmp_path: Path) -> None:
    assert load_people(tmp_path / "missing" / "people.json") == PeopleList()


def test_the_meetings_vocabulary_is_folded_in_once_and_the_file_is_left_as_it_was(tmp_path: Path) -> None:
    path = _saved(tmp_path / "people.json", Term("Northwind", note="the client"))
    vocabulary = tmp_path / "Meetings" / "vocabulary.txt"
    vocabulary.parent.mkdir()
    original = "# Names and terms\n\nLaunch Board\n  BRIGHTLINE  \nnorthwind\nLaunch Board\nQuoteDesk\n"
    vocabulary.write_text(original, encoding="utf-8")

    import_vocabulary(path, vocabulary)
    vocabulary.write_text(original + "Added Later\n", encoding="utf-8")
    import_vocabulary(path, vocabulary)

    assert [term.written for term in load_people(path).terms] == ["Northwind", "Launch Board", "BRIGHTLINE", "QuoteDesk"]
    assert load_people(path).terms[0].note == "the client"
    assert vocabulary.read_text(encoding="utf-8") == original + "Added Later\n"


def test_without_a_vocabulary_file_the_import_is_done_with_nothing_added(tmp_path: Path) -> None:
    path = tmp_path / "people.json"
    import_vocabulary(path, tmp_path / "Meetings" / "vocabulary.txt")
    assert load_people(path) == PeopleList(vocabulary_imported=True)


def test_a_vocabulary_file_that_cannot_be_read_is_tried_again_next_time(tmp_path: Path) -> None:
    path = tmp_path / "people.json"
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_bytes("Zürich\n".encode("utf-16"))

    import_vocabulary(path, vocabulary)
    assert not load_people(path).vocabulary_imported

    vocabulary.write_text("Zürich\n", encoding="utf-8")
    import_vocabulary(path, vocabulary)
    assert load_people(path).terms == (Term("Zürich"),)


def test_each_entry_with_heard_as_spellings_becomes_one_swap_rule(tmp_path: Path) -> None:
    rules = people_rules((
        Term("Skryb", note="the app", heard_as=("Skribe", "Scryb")),
        Term("Slack"),
        Term("Ngozi", phonetic=True),
        Term("Odile", heard_as=("Odeel",), phonetic=True),
    ))

    assert rules == (
        LiteralRule(spoken=("Skribe", "Scryb"), replacement="Skryb"),
        LiteralRule(spoken=("Ngozi",), replacement="Ngozi", case_sensitive=True, phonetic=True),
        LiteralRule(spoken=("Odeel",), replacement="Odile", phonetic=True),
    )


def test_the_same_re_pick_counts_up_and_a_later_one_for_the_same_spelling_replaces_it(tmp_path: Path) -> None:
    path = _saved(tmp_path / "people.json", Term("Mikaela"), Term("Michaela"))

    suggest(path, "Mikala", "Mikaela")
    suggest(path, "mikala", "Mikaela")
    suggest(path, "Rafe", "Mikaela")
    assert load_people(path).suggestions == (Suggestion("mikala", "Mikaela", 2), Suggestion("Rafe", "Mikaela", 1))

    suggest(path, "Mikala", "Michaela")
    assert load_people(path).suggestions == (Suggestion("Rafe", "Mikaela", 1), Suggestion("Mikala", "Michaela", 1))


@pytest.mark.parametrize(("heard", "written"), [
    ("Michaela", "Mikaela"),
    ("MIKAYLA", "Mikaela"),
    ("scribe", "Skryb"),
    ("Mikala", "Nobody Listed"),
])
def test_a_spelling_that_is_an_entry_already_swapped_dismissed_or_for_no_entry_is_not_suggested(
    tmp_path: Path, heard: str, written: str,
) -> None:
    path = _saved(tmp_path / "people.json", Term("Mikaela", heard_as=("Mikayla",)), Term("Michaela"), Term("Skryb"))
    update_people(path, lambda people: replace(people, dismissed=(("scribe", "Skryb"),)))

    suggest(path, heard, written)

    assert load_people(path).suggestions == ()


def test_accepting_adds_the_spelling_to_its_entry_and_dismissing_remembers_it(tmp_path: Path) -> None:
    people = PeopleList(
        terms=(Term("Skryb", heard_as=("Skribe",)), Term("Odile")),
        suggestions=(Suggestion("Scryb", "Skryb", 3), Suggestion("Odeel", "Odile")),
    )

    accepted = accept_suggestion(people, " scryb ")
    dismissed = dismiss_suggestion(accepted, "ODEEL")

    assert dismissed.terms == (Term("Skryb", heard_as=("Skribe", "Scryb")), Term("Odile"))
    assert dismissed.suggestions == ()
    assert dismissed.dismissed == (("odeel", "Odile"),)
    with pytest.raises(KeyError):
        accept_suggestion(dismissed, "Odeel")


def test_saving_new_entries_drops_the_suggestions_for_entries_that_are_gone() -> None:
    people = PeopleList(
        terms=(Term("Skryb"), Term("Odile")),
        suggestions=(Suggestion("Scryb", "Skryb"), Suggestion("Odeel", "Odile")),
    )

    assert with_terms(people, (Term("Skryb"),)).suggestions == (Suggestion("Scryb", "Skryb"),)


def test_thousands_of_entries_save_and_load_whole(tmp_path: Path) -> None:
    terms = tuple(Term(f"Name {index}", heard_as=(f"Nayme {index}",)) for index in range(3000))
    path = _saved(tmp_path / "people.json", *terms)
    assert load_people(path).terms == terms


def test_re_picks_from_many_threads_at_once_are_all_kept(tmp_path: Path) -> None:
    path = _saved(tmp_path / "people.json", Term("Skryb"))
    threads = [threading.Thread(target=suggest, args=(path, f"Skrib{index}", "Skryb")) for index in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(load_people(path).suggestions) == 40


def test_entries_from_the_settings_window_are_trimmed_and_their_duplicate_spellings_merged() -> None:
    terms = terms_from_json([
        {"written": "  Skryb ", "note": " the   app ", "heard_as": ["Skribe", " skribe ", "", "SKRYB"], "phonetic": False},
    ])
    assert terms == (Term("Skryb", note="the app", heard_as=("Skribe",)),)
