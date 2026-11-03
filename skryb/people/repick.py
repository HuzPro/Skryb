"""Dictating over a selection that already says what was dictated picks the other spelling.

Selecting a name the engine got wrong (a double click takes the word and the
space after it) and saying it again asks for a different spelling, not the
same one back. The spellings to cycle through are the entries that sound
like the selection, two people called Mikaela and Michaela say, and the
ordinary word when the entry is a name like "Skryb" that is also "scribe".
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jellyfish

from skryb.people.terms import Term, suggest

_logger = logging.getLogger(__name__)

_MOST_WORDS = 4
# The share of letters two spellings may differ by and still be one name said twice.
_SOUND_DISTANCE = 0.34
_EDGES = " \t\r\n.,;:!?\"“”()[]{}…"
_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)


@dataclass(frozen=True)
class Repick:
    text: str
    # The spellings that were dictated over, to suggest as heard-as when *written* is an entry.
    heard: tuple[str, ...]
    written: str | None


def repick(text: str, selected: str, terms: Sequence[Term]) -> Repick | None:
    """*text* with the next spelling after the selection's, or None when it says something else."""
    said, chosen = text.strip(_EDGES), selected.strip(_EDGES)
    if not said or not chosen or len(chosen.split()) > _MOST_WORDS:
        return None
    said_base, tail = _split_possessive(said)
    chosen_base, chosen_tail = _split_possessive(chosen)
    group = [term for term in terms if _covers(term, chosen_base)]
    if tail.casefold() != chosen_tail.casefold() or not group:
        return None
    if not (_same(said_base, chosen_base) or any(_covers(term, said_base) for term in group)):
        return None
    names = [term.written for term in group]
    # Two people with one name cycle between them; one entry also gives back the ordinary word it is heard as.
    others = (chosen_base, said_base, *group[0].heard_as) if len(names) == 1 else (chosen_base,)
    spellings = _distinct(names + [word for word in others if not any(_same(word, term.written) for term in terms)])
    if len(spellings) < 2:
        return None
    at = next(index for index, spelling in enumerate(spellings) if _same(spelling, chosen_base))
    pick = spellings[(at + 1) % len(spellings)]
    is_name = pick in names
    start = text.index(said)
    changed = text[:start] + (pick if is_name else _ordinary_case(pick)) + tail + text[start + len(said):]
    heard = tuple(word for word in _distinct([said_base, chosen_base]) if not _same(word, pick))
    return Repick(changed, heard, pick if is_name else None)


def suggest_heard_as(people: Path, picked: Repick) -> None:
    """Offer what was dictated over as a heard-as spelling of the entry picked instead."""
    _logger.info("Dictating over a selection picked another spelling.")
    if picked.written:
        for heard in picked.heard:
            suggest(people, heard, picked.written)


def keep_written_case(text: str, written: str) -> str:
    """*text* with *written* spelt as the entry is, after fitting it mid-sentence lowered or raised its first letter."""
    return re.sub(rf"(?<!\w){re.escape(written)}(?!\w)", lambda _match: written, text, count=1, flags=re.IGNORECASE)


def _distinct(words: list[str]) -> list[str]:
    kept: list[str] = []
    for word in words:
        if not any(_same(word, other) for other in kept):
            kept.append(word)
    return kept


def _covers(term: Term, word: str) -> bool:
    return (
        _same(word, term.written)
        or any(_same(word, spelling) for spelling in term.heard_as)
        or _sounds_alike(word, term.written)
    )


def _same(first: str, second: str) -> bool:
    return _squashed(first) == _squashed(second)


def _sounds_alike(first: str, second: str) -> bool:
    a, b = _squashed(first), _squashed(second)
    if not a or not b:
        return False
    if jellyfish.metaphone(a) == jellyfish.metaphone(b):
        return True
    return jellyfish.damerau_levenshtein_distance(a, b) / max(len(a), len(b)) <= _SOUND_DISTANCE


def _squashed(text: str) -> str:
    return "".join(text.split()).casefold()


def _split_possessive(word: str) -> tuple[str, str]:
    match = _POSSESSIVE.search(word)
    if match is None or match.start() == 0:
        return word, ""
    return word[:match.start()], word[match.start():]


def _ordinary_case(word: str) -> str:
    """The ordinary word in lower case, for the caret to capitalise at a sentence start; acronyms keep theirs."""
    if word == "I" or any(character.isupper() for character in word[1:]):
        return word
    return word[:1].lower() + word[1:]
