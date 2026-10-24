"""Learns spellings and style from the edits made to pasted dictations.

Fed by edit capture's ``sent`` events. Only reliable, unflagged edits count,
and only word-level facts are kept: which spoken word became which written
one, and three style tallies. No dictated or typed text is stored.
"""

import json
import logging
import re
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import cast

from skryb.output.dictionary_config import (
    DictionaryConfig,
    LiteralRule,
    update_dictionary,
)
from skryb.writing.settings import WritingSettings

_logger = logging.getLogger(__name__)

_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*")
_CONTRACTION = re.compile(r"\b\w+['’](?:t|s|re|ve|ll|d|m)\b", re.IGNORECASE)
_MAX_PHRASE_WORDS = 3
_MIN_SIMILARITY = 0.5
DICTATIONS_TO_TEACH = 2
MAX_CANDIDATES = 500
MAX_BLOCKED = 1000
_MAX_SOURCES = 4
STYLE_THRESHOLD = 3
_STYLE_LIMIT = 6

FUNCTION_WORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself let's me more most mustn't my myself no nor not of off on once only or
other ought our ours ourselves out over own same shan't she she'd she'll she's
should shouldn't so some such than that that's the their theirs them themselves
then there there's these they they'd they'll they're they've this those through
to too under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())

# signal: (sentence when positive, sentence when negative,
#          summary when positive, summary when negative)
_STYLE_SIGNALS: dict[str, tuple[str, str, str, str]] = {
    "final_stop": (
        "Do not end the message with a full stop.",
        "End the message with a full stop.",
        "drops the final full stop",
        "ends with a full stop",
    ),
    "contractions": (
        "Use contractions (don't, it's).",
        "Do not use contractions (write do not, it is).",
        "prefers contractions",
        "avoids contractions",
    ),
    "lowercase_start": (
        "Start the message with a lowercase letter.",
        "Start the message with a capital letter.",
        "starts in lowercase",
        "starts with a capital",
    ),
}


@dataclass(frozen=True)
class WordCorrection:
    spoken: str
    written: str

    @property
    def key(self) -> tuple[str, str]:
        return self.spoken.casefold(), self.written


@dataclass
class _Candidate:
    spoken: str
    written: str
    sources: list[str] = field(default_factory=list)
    seen: int = 0


def word_corrections(pasted: str, final: str) -> list[WordCorrection]:
    """The spelling or name fixes between what was pasted and what was sent."""
    before = _WORD.findall(pasted)
    after = _WORD.findall(final)
    found: list[WordCorrection] = []
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, start, end, new_start, new_end in matcher.get_opcodes():
        if tag != "replace":
            continue
        old, new = before[start:end], after[new_start:new_end]
        if len(old) > _MAX_PHRASE_WORDS or len(new) > _MAX_PHRASE_WORDS:
            continue
        correction = WordCorrection(" ".join(old), " ".join(new))
        if _is_spelling_fix(old, correction):
            found.append(correction)
    return found


def _is_spelling_fix(old: list[str], correction: WordCorrection) -> bool:
    if not (_is_word(correction.spoken) and _is_word(correction.written)):
        return False
    if any(word.casefold() in FUNCTION_WORDS for word in old):
        return False
    similarity = SequenceMatcher(
        a=correction.spoken.casefold(), b=correction.written.casefold(),
    ).ratio()
    return similarity >= _MIN_SIMILARITY


def _is_word(text: str) -> bool:
    return bool(text) and all(
        character.isalpha() or character in " '’-" for character in text
    )


def style_votes(pasted: str, final: str) -> dict[str, int]:
    """+1 or -1 for each style signal this one edit shows."""
    votes: dict[str, int] = {}
    before, after = pasted.strip(), final.strip()
    before_words, after_words = _WORD.findall(before), _WORD.findall(after)
    if not before_words or not after_words:
        return votes
    if before_words[-1] == after_words[-1]:
        if before.endswith(".") and not after.endswith("."):
            votes["final_stop"] = 1
        elif after.endswith(".") and not before.endswith("."):
            votes["final_stop"] = -1
    contracted = len(_CONTRACTION.findall(after)) - len(_CONTRACTION.findall(before))
    if contracted:
        votes["contractions"] = 1 if contracted > 0 else -1
    first, first_after = before_words[0], after_words[0]
    if first.casefold() == first_after.casefold() and first != first_after:
        if first[0].isupper() and first_after[0].islower():
            votes["lowercase_start"] = 1
        elif first[0].islower() and first_after[0].isupper():
            votes["lowercase_start"] = -1
    return votes


class Learner:
    """Turns repeated corrections into dictionary words and style notes."""

    def __init__(
        self,
        path: Path,
        dictionary_path: Path,
        settings: Callable[[], WritingSettings],
        on_dictionary_changed: Callable[[], None],
        dictation_prefix: str,
    ) -> None:
        self._path = path
        self._dictionary_path = dictionary_path
        self._settings = settings
        self._on_dictionary_changed = on_dictionary_changed
        self._prefix = dictation_prefix
        self._lock = threading.Lock()
        self._loaded = False
        self._candidates: dict[tuple[str, str], _Candidate] = {}
        self._blocked: dict[tuple[str, str], None] = {}
        self._style: dict[str, int] = {}
        self._tick = 0

    def observe(self, event: dict[str, object]) -> None:
        """An edit-capture event; only reliable, unflagged ``sent`` ones count."""
        if event.get("event") != "sent" or event.get("reliable") is not True:
            return
        if event.get("flags"):
            return
        settings = self._settings()
        if not settings.learns:
            return
        spans = event.get("spans")
        if not isinstance(spans, list):
            return
        with self._lock:
            self._ensure_loaded()
            ready = self._count(cast(list[object], spans), settings)
            self._save()
        if ready:
            self._teach(ready)

    def observe_correction(self, pasted: str, final: str, dictation: str) -> None:
        """One fix seen outside edit capture: a dictation's words as pasted and as they ended up.

        ``dictation`` names the fix, so the same pair counts once per fix and
        is taught only when a second, separate fix agrees.
        """
        if not self._settings().learn_words:
            return
        with self._lock:
            self._ensure_loaded()
            ready = self._count_corrections(pasted, final, f"{self._prefix}:{dictation}")
            self._save()
        if ready:
            self._teach(ready)

    def style_sentences(self) -> tuple[str, ...]:
        with self._lock:
            self._ensure_loaded()
            return tuple(
                sentences[0] if self._style[name] > 0 else sentences[1]
                for name, sentences in _STYLE_SIGNALS.items()
                if abs(self._style.get(name, 0)) >= STYLE_THRESHOLD
            )

    def style_summary(self) -> str:
        with self._lock:
            self._ensure_loaded()
            learned = [
                labels[2] if self._style[name] > 0 else labels[3]
                for name, labels in _STYLE_SIGNALS.items()
                if abs(self._style.get(name, 0)) >= STYLE_THRESHOLD
            ]
        return "; ".join(learned) if learned else "nothing yet"

    def forget_style(self) -> None:
        with self._lock:
            self._ensure_loaded()
            self._style = {}
            self._save()

    def reset(self) -> None:
        with self._lock:
            self._loaded = True
            self._candidates = {}
            self._blocked = {}
            self._style = {}
            self._save()

    def block(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Never teach these ``(spoken, written)`` pairs again, until a reset."""
        with self._lock:
            self._ensure_loaded()
            for spoken, written in pairs:
                key = (spoken.strip().casefold(), written)
                self._candidates.pop(key, None)
                self._blocked.pop(key, None)
                self._blocked[key] = None
            while len(self._blocked) > MAX_BLOCKED:
                del self._blocked[next(iter(self._blocked))]
            self._save()

    def _count(
        self, spans: list[object], settings: WritingSettings,
    ) -> list[WordCorrection]:
        ready: list[WordCorrection] = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            fields = cast(dict[str, object], span)
            pasted, final, seq = fields.get("pasted_text"), fields.get("final_text"), fields.get("seq")
            if not isinstance(pasted, str) or not isinstance(final, str) or pasted == final:
                continue
            if settings.learn_words:
                ready.extend(self._count_corrections(pasted, final, f"{self._prefix}:{seq}"))
            if settings.learn_style:
                self._count_style(pasted, final)
        return ready

    def _count_corrections(
        self, pasted: str, final: str, dictation: str,
    ) -> list[WordCorrection]:
        ready: list[WordCorrection] = []
        for correction in {c.key: c for c in word_corrections(pasted, final)}.values():
            if correction.key in self._blocked:
                continue
            candidate = self._candidates.get(correction.key)
            if candidate is None:
                candidate = _Candidate(correction.spoken, correction.written)
                self._candidates[correction.key] = candidate
            self._tick += 1
            candidate.seen = self._tick
            if dictation not in candidate.sources:
                candidate.sources = (candidate.sources + [dictation])[-_MAX_SOURCES:]
            if len(candidate.sources) >= DICTATIONS_TO_TEACH:
                ready.append(correction)
        self._evict_stale_candidates()
        return ready

    def _evict_stale_candidates(self) -> None:
        excess = len(self._candidates) - MAX_CANDIDATES
        if excess <= 0:
            return
        stale = sorted(self._candidates, key=lambda key: self._candidates[key].seen)
        for key in stale[:excess]:
            del self._candidates[key]

    def _count_style(self, pasted: str, final: str) -> None:
        for name, vote in style_votes(pasted, final).items():
            tally = self._style.get(name, 0) + vote
            self._style[name] = max(-_STYLE_LIMIT, min(_STYLE_LIMIT, tally))

    def _teach(self, corrections: list[WordCorrection]) -> None:
        def add_learned_words(dictionary: DictionaryConfig) -> DictionaryConfig:
            used = {
                alias.strip().casefold()
                for rule in dictionary.rules
                if isinstance(rule, LiteralRule)
                for alias in rule.spoken
            }
            rules = list(dictionary.rules)
            for correction in corrections:
                if correction.spoken.casefold() in used:
                    continue
                used.add(correction.spoken.casefold())
                rules.append(LiteralRule(
                    spoken=(correction.spoken,),
                    replacement=correction.written,
                    learned=True,
                ))
            return replace(dictionary, rules=tuple(rules))

        if update_dictionary(self._dictionary_path, add_learned_words):
            _logger.info("Learned %d word(s) from corrections.", len(corrections))
            self._on_dictionary_changed()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            self._load()
        except (OSError, UnicodeError, ValueError, TypeError, KeyError, IndexError) as exc:
            _logger.warning("Learning data ignored: %s", exc)
            self._candidates, self._blocked, self._style = {}, {}, {}

    def _load(self) -> None:
        if not self._path.exists():
            return
        document: object = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("learning data must be a JSON object")
        self._apply(cast(dict[str, object], document))

    def _apply(self, document: dict[str, object]) -> None:
        for raw in cast(list[dict[str, object]], document.get("candidates", [])):
            candidate = _Candidate(
                spoken=str(raw["spoken"]),
                written=str(raw["written"]),
                sources=[str(source) for source in cast(list[object], raw.get("sources", []))],
                seen=int(cast(int, raw.get("seen", 0))),
            )
            self._candidates[(candidate.spoken.casefold(), candidate.written)] = candidate
        for pair in cast(list[list[str]], document.get("blocked", [])):
            self._blocked[(str(pair[0]), str(pair[1]))] = None
        style = cast(dict[str, object], document.get("style", {}))
        self._style = {
            name: int(cast(int, style[name])) for name in _STYLE_SIGNALS if name in style
        }
        self._tick = max((c.seen for c in self._candidates.values()), default=0)

    def _save(self) -> None:
        try:
            self._write()
        except OSError as exc:
            _logger.warning("Could not save learning data: %s", exc)

    def _write(self) -> None:
        document = {
            "version": 1,
            "candidates": [
                {"spoken": c.spoken, "written": c.written, "sources": c.sources, "seen": c.seen}
                for c in self._candidates.values()
            ],
            "blocked": [list(key) for key in self._blocked],
            "style": self._style,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        staged = self._path.with_suffix(".tmp")
        staged.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        staged.replace(self._path)
