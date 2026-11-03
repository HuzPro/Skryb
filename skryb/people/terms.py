"""The people and terms list: the names and words dictation and meetings should spell right.

One list serves both. Each entry is how the name is written, an optional
note that tells the model which thing it is ("the app I'm building") and the
spellings it has been heard as, which are swapped for it. ``people.json`` also
keeps the swaps that re-dictating over a selection suggested, until each is
accepted or dismissed.
"""

import json
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from skryb.output.dictionary_config import LiteralRule

_logger = logging.getLogger(__name__)
# Dictation's re-picks and the settings window both rewrite people.json, from different threads.
_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Term:
    written: str
    note: str = ""
    heard_as: tuple[str, ...] = ()
    # Also swap words that only sound like the entry, not just the spellings listed.
    phonetic: bool = False


@dataclass(frozen=True)
class Suggestion:
    """A spelling to swap for *written*, from re-dictating over a selection that held *heard*."""

    heard: str
    written: str
    times: int = 1


@dataclass(frozen=True)
class PeopleList:
    terms: tuple[Term, ...] = ()
    suggestions: tuple[Suggestion, ...] = ()
    # (heard casefolded, written) pairs never to suggest again.
    dismissed: tuple[tuple[str, str], ...] = ()
    vocabulary_imported: bool = False


def load_people(path: Path) -> PeopleList:
    """The saved list; a part that cannot be read is skipped rather than losing the rest."""
    try:
        document: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return PeopleList()
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _logger.warning("People and terms list ignored: %s", exc)
        return PeopleList()
    if not isinstance(document, dict):
        _logger.warning("People and terms list ignored: it is not a JSON object")
        return PeopleList()
    raw = cast(dict[str, object], document)
    return PeopleList(
        terms=_tidy([term for row in _rows(raw, "people") if (term := _term(row)) is not None]),
        suggestions=tuple(s for row in _rows(raw, "suggestions") if (s := _suggestion(row)) is not None),
        dismissed=tuple(
            (row[0], row[1]) for row in _rows(raw, "dismissed")
            if isinstance(row, list) and len(row) == 2 and all(isinstance(part, str) for part in row)
        ),
        vocabulary_imported=raw.get("vocabulary_imported") is True,
    )


def load_terms(path: Path) -> tuple[Term, ...]:
    return load_people(path).terms


def update_people(path: Path, change: Callable[[PeopleList], PeopleList]) -> PeopleList:
    """Apply *change* to the saved list and save it when it changed; returns the list as it now stands."""
    with _WRITE_LOCK:
        current = load_people(path)
        updated = change(current)
        if updated != current:
            _write(path, updated)
        return updated


def _write(path: Path, people: PeopleList) -> None:
    document = {
        "version": 1,
        "people": [term_json(term) for term in people.terms],
        "suggestions": [suggestion_json(s) for s in people.suggestions],
        "dismissed": [list(pair) for pair in people.dismissed],
        "vocabulary_imported": people.vocabulary_imported,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    staged.replace(path)


def term_json(term: Term) -> dict[str, object]:
    return {"written": term.written, "note": term.note, "heard_as": list(term.heard_as), "phonetic": term.phonetic}


def suggestion_json(suggestion: Suggestion) -> dict[str, object]:
    return {"heard": suggestion.heard, "written": suggestion.written, "times": suggestion.times}


def terms_from_json(rows: list[object]) -> tuple[Term, ...]:
    """The entries the settings window sent; raises ``ValueError`` naming the first that cannot be saved."""
    terms: list[Term] = []
    for index, row in enumerate(rows):
        term = _term(row)
        if term is None:
            raise ValueError(f"Entry {index + 1} needs how it is written, a note, heard-as spellings and sounds-alike.")
        terms.append(term)
    problem = _first_conflict(terms)
    if problem is not None:
        raise ValueError(problem)
    return tuple(terms)


def people_rules(terms: Iterable[Term]) -> tuple[LiteralRule, ...]:
    """Each entry's heard-as spellings as one swap rule; a sounds-alike entry with none matches by sound alone."""
    rules: list[LiteralRule] = []
    for term in terms:
        if term.heard_as:
            rules.append(LiteralRule(spoken=term.heard_as, replacement=term.written, phonetic=term.phonetic))
        elif term.phonetic:
            # Its own exact spelling gives the sound match a target and is left as it is.
            rules.append(LiteralRule(spoken=(term.written,), replacement=term.written, case_sensitive=True, phonetic=True))
    return tuple(rules)


def import_vocabulary(path: Path, vocabulary: Path) -> None:
    """Fold the meetings' old ``vocabulary.txt`` into the list, once. The file itself is left as it was."""
    if load_people(path).vocabulary_imported:
        return
    try:
        lines = vocabulary.read_text(encoding="utf-8").splitlines() if vocabulary.exists() else []
    except (OSError, UnicodeDecodeError) as exc:
        _logger.warning("The meetings' vocabulary.txt could not be read, so it is not in the list yet: %s", exc)
        return
    written = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]

    def add_vocabulary(people: PeopleList) -> PeopleList:
        if people.vocabulary_imported:
            return people
        known = {term.written.casefold() for term in people.terms}
        added = tuple(Term(word) for word in dict.fromkeys(written) if word.casefold() not in known)
        return replace(people, terms=_tidy(people.terms + added), vocabulary_imported=True)

    update_people(path, add_vocabulary)
    if written:
        _logger.info("Moved %d names and terms from %s into the people and terms list.", len(written), vocabulary)


def suggest(path: Path, heard: str, written: str) -> None:
    """Remember that a selection holding *heard* was re-dictated into *written*."""
    def add_suggestion(people: PeopleList) -> PeopleList:
        folded = heard.casefold()
        term = next((t for t in people.terms if t.written == written), None)
        if term is None or (folded, written) in people.dismissed:
            return people
        if folded in {h.casefold() for h in term.heard_as} or folded in {t.written.casefold() for t in people.terms}:
            return people
        earlier = next((s for s in people.suggestions if s.heard.casefold() == folded), None)
        times = earlier.times + 1 if earlier is not None and earlier.written == written else 1
        others = tuple(s for s in people.suggestions if s is not earlier)
        return replace(people, suggestions=others + (Suggestion(heard, written, times),))

    update_people(path, add_suggestion)


def accept_suggestion(people: PeopleList, heard: str) -> PeopleList:
    """*heard* added to its entry's heard-as spellings. Raises ``KeyError`` when there is no such suggestion."""
    suggestion = _suggestion_for(people, heard)
    terms = tuple(
        replace(term, heard_as=term.heard_as + (suggestion.heard,)) if term.written == suggestion.written else term
        for term in people.terms
    )
    return replace(people, terms=_tidy(terms), suggestions=_without(people, suggestion))


def dismiss_suggestion(people: PeopleList, heard: str) -> PeopleList:
    """Drop the suggestion for good. Raises ``KeyError`` when there is no such suggestion."""
    suggestion = _suggestion_for(people, heard)
    pair = (suggestion.heard.casefold(), suggestion.written)
    return replace(people, suggestions=_without(people, suggestion), dismissed=people.dismissed + (pair,))


def with_terms(people: PeopleList, terms: tuple[Term, ...]) -> PeopleList:
    """*terms* in place of the saved entries; suggestions for an entry that is gone go with it."""
    kept = {term.written for term in terms}
    return replace(people, terms=terms, suggestions=tuple(s for s in people.suggestions if s.written in kept))


def _suggestion_for(people: PeopleList, heard: str) -> Suggestion:
    folded = heard.strip().casefold()
    for suggestion in people.suggestions:
        if suggestion.heard.casefold() == folded:
            return suggestion
    raise KeyError(heard)


def _without(people: PeopleList, suggestion: Suggestion) -> tuple[Suggestion, ...]:
    return tuple(s for s in people.suggestions if s is not suggestion)


def _tidy(terms: Iterable[Term]) -> tuple[Term, ...]:
    """Keep the first entry for each spelling, and drop heard-as spellings that are another entry or already taken.

    Swapping one entry's name for another's would erase that person, and a
    spelling under two entries could go to either.
    """
    unique: dict[str, Term] = {}
    for term in terms:
        unique.setdefault(term.written.casefold(), term)
    taken = set(unique)
    tidy: list[Term] = []
    for term in unique.values():
        heard: list[str] = []
        for spelling in term.heard_as:
            if spelling.casefold() not in taken:
                taken.add(spelling.casefold())
                heard.append(spelling)
        tidy.append(replace(term, heard_as=tuple(heard)))
    return tuple(tidy)


def _first_conflict(terms: list[Term]) -> str | None:
    names: set[str] = set()
    for term in terms:
        if term.written.casefold() in names:
            return f"“{term.written}” is in the list twice."
        names.add(term.written.casefold())
    owners: dict[str, str] = {}
    for term in terms:
        for spelling in term.heard_as:
            if spelling.casefold() in names:
                return f"“{spelling}” is an entry of its own, so it can't be swapped for “{term.written}”."
            if spelling.casefold() in owners:
                return f"“{spelling}” is heard as both “{owners[spelling.casefold()]}” and “{term.written}”."
            owners[spelling.casefold()] = term.written
    return None


def _rows(document: dict[str, object], key: str) -> list[object]:
    value = document.get(key)
    return cast(list[object], value) if isinstance(value, list) else []


def _term(row: object) -> Term | None:
    if not isinstance(row, dict):
        return None
    fields = cast(dict[str, object], row)
    written, note, heard, phonetic = (
        fields.get("written"), fields.get("note", ""), fields.get("heard_as", []), fields.get("phonetic", False),
    )
    if not isinstance(written, str) or not written.strip() or not isinstance(note, str) or not isinstance(phonetic, bool):
        return None
    if not isinstance(heard, list) or not all(isinstance(spelling, str) for spelling in heard):
        return None
    spellings: dict[str, str] = {written.strip().casefold(): ""}
    for spelling in (s.strip() for s in cast(list[str], heard)):
        if spelling:
            spellings.setdefault(spelling.casefold(), spelling)
    return Term(written.strip(), " ".join(note.split()), tuple(s for s in spellings.values() if s), phonetic)


def _suggestion(row: object) -> Suggestion | None:
    if not isinstance(row, dict):
        return None
    fields = cast(dict[str, object], row)
    heard, written, times = fields.get("heard"), fields.get("written"), fields.get("times", 1)
    if not isinstance(heard, str) or not isinstance(written, str) or not heard.strip() or not written.strip():
        return None
    if isinstance(times, bool) or not isinstance(times, int) or times < 1:
        times = 1
    return Suggestion(heard.strip(), written.strip(), times)
