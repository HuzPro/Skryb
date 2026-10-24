"""The writing model as one stage of the batch text pipeline.

Every way this stage can fail ends the same way: the dictation goes on
exactly as it came in, so cleanup can only ever add polish, never lose text.
"""

import logging
import re
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from typing import Protocol

from skryb.edit_capture.recorder import DENIED_PROCESSES
from skryb.edit_capture.window import ForegroundWindow
from skryb.output.caret_context import NO_CONTEXT, CaretContext
from skryb.output.dictation_trace import (
    APPLIED,
    REJECTED,
    SKIPPED,
    TOO_SLOW,
    DictationTrace,
    current_trace,
)
from skryb.writing.client import WritingUnavailable
from skryb.writing.prompt import (
    Hints,
    cleanup_system_prompt,
    cleanup_user_message,
    estimated_tokens,
)
from skryb.writing.settings import WritingSettings

_logger = logging.getLogger(__name__)

CONTEXT_TOKENS = 8192
DEFAULT_SECONDS_PER_100_WORDS = 1.6
# Conversational dictation runs at about 150 words a minute.
_SPOKEN_WORDS_PER_SECOND = 2.5
_TIMINGS_KEPT = 20

_MIN_LENGTH_RATIO = 0.4
_MAX_LENGTH_RATIO = 2.5
_NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")
# A number said just before a correction is the one being taken back, so it
# is the only number cleanup may drop: "12, sorry, 14 dollars".
_CORRECTED_NUMBER = re.compile(
    r"(?P<old>\d+(?:[.,:/]\d+)*)(?:\s+[^\W\d_]+){0,2}[\s,;.]*"
    r"(?:actually|sorry|no wait|wait no|no|make (?:that|it)|i mean|rather|"
    r"correction|scratch that)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(r"\b(?:not|never|cannot|without)\b|n't\b", re.IGNORECASE)
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
# The model's only jobs are a spoken correction and a false start. Of 302
# real dictations, under 30 had one; calling the model on the rest cost
# 0.4 s each and gave nothing but rewrites ("According to the." became
# "According to God"). Without a cue the dictation is pasted as recognised.
# Spoken lists are not a job: every list the model laid out from a real
# dictation was wrong, so a list in its answer is rejected below.
# "make it" is left out: it is almost always an instruction to someone ("make
# it async"), and 16 of 17 dictations it sent to the model came back unchanged.
_CUES = re.compile(
    r"\b(?:actually|no wait|wait no|i mean|make that|scratch that|strike that|"
    r"never ?mind|sorry|correction)\b"
    # a restarted phrase: "the gondola needs, the gondola needs a rope"
    r"|\b((?:[^\W\d_]+ ){1,3}[^\W\d_]+)[,.]? \1\b",
    re.IGNORECASE,
)
_MIN_WORDS = 6
NO_CUE = "no spoken correction, false start or list"
# Words that soften a claim. "I was thinking" is left out: as a lead-in
# ("so I was thinking we should...") it is filler, not doubt.
_HEDGES = ("maybe", "perhaps", "probably", "might", "not sure", "i think", "i guess")
_PREAMBLES = re.compile(r"(?:sure|here)\b", re.IGNORECASE)
_QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("'", "'"))


class CompletionClient(Protocol):
    def complete(
        self, system: str, user: str, max_tokens: int, timeout: float,
    ) -> str: ...


class CleanupTimings:
    """How long recent cleanups took, for the pill's estimate and the status line.

    The rate is pooled over recent successes, so long dictations, where the
    per-word cost dominates, weigh more than short ones.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recent: deque[tuple[float, int]] = deque(maxlen=_TIMINGS_KEPT)

    def record(self, seconds: float, words: int) -> None:
        with self._lock:
            self._recent.append((seconds, max(words, 1)))

    def last_seconds(self) -> float | None:
        with self._lock:
            return self._recent[-1][0] if self._recent else None

    def seconds_per_100_words(self) -> float:
        with self._lock:
            words = sum(count for _, count in self._recent)
            seconds = sum(taken for taken, _ in self._recent)
        return 100 * seconds / words if words else DEFAULT_SECONDS_PER_100_WORDS

    def estimate_for_audio(self, audio_seconds: float) -> float:
        words = audio_seconds * _SPOKEN_WORDS_PER_SECOND
        return self.seconds_per_100_words() * words / 100


class CleanupStage:
    """Asks the writing model to tidy a dictation, within a deadline."""

    def __init__(
        self,
        settings: WritingSettings,
        client: CompletionClient,
        timings: CleanupTimings,
        spellings: tuple[str, ...] = (),
        learned_style: Callable[[], tuple[str, ...]] = tuple,
        transform: Callable[[], str] = str,
        window: Callable[[], ForegroundWindow] = ForegroundWindow,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings
        self._client = client
        self._timings = timings
        self._spellings = spellings
        self._learned_style = learned_style
        self._transform = transform
        self._window = window
        self._clock = clock

    def __call__(self, text: str, caret_context: Callable[[], CaretContext]) -> str:
        if self._settings.cleanup == "off" or not text.strip():
            return text
        started = self._clock()
        transform = self._transform()
        dictation = text.strip()
        trace = current_trace() or DictationTrace()
        if not transform.strip() and not needs_cleanup(dictation):
            return _outcome(text, trace, SKIPPED, NO_CUE)
        window = self._window() if self._settings.context else None
        context = (
            caret_context()
            if window is not None and window.process not in DENIED_PROCESSES
            else NO_CONTEXT
        )
        system = cleanup_system_prompt(self._settings, self._style(), transform)
        hints = self._hints(window, context)
        trace.context_before, trace.window = hints.context_before, hints.window
        user = cleanup_user_message(dictation)
        max_tokens = estimated_tokens(dictation) * 2 + 64
        if estimated_tokens(system + user) + max_tokens > CONTEXT_TOKENS:
            return _outcome(text, trace, SKIPPED, "the dictation is too long for the writing model")
        deadline = self._settings.deadline_seconds
        try:
            answer = self._client.complete(
                system, user, max_tokens, deadline - (self._clock() - started),
            )
        except WritingUnavailable as exc:
            return _outcome(text, trace, SKIPPED, str(exc))
        elapsed = self._clock() - started
        trace.cleanup_answer, trace.cleanup_seconds = answer, elapsed
        if elapsed > deadline:
            return _outcome(
                text, trace, TOO_SLOW, f"it took {elapsed:.2f}s, over the {deadline:.1f}s limit",
            )
        reason = rejection_reason(dictation, answer, self._settings.cleanup, bool(transform.strip()))
        if reason:
            return _outcome(text, trace, REJECTED, reason)
        self._timings.record(elapsed, len(dictation.split()))
        _logger.info("Cleanup applied in %.2fs.", elapsed)
        trace.cleanup = APPLIED
        return _with_edges_of(text, answer.strip())

    def _style(self) -> tuple[str, ...]:
        notes = self._settings.style_notes.strip()
        learned = self._learned_style() if self._settings.learn_style else ()
        return ((notes,) if notes else ()) + learned

    def _hints(self, window: ForegroundWindow | None, context: CaretContext) -> Hints:
        """Spellings always; the caret and window only when allowed and not a denied app."""
        if window is None or window.process in DENIED_PROCESSES:
            return Hints(spellings=self._spellings)
        title = f"{window.title} ({window.process})" if window.process else window.title
        return Hints(self._spellings, context.before, title)


def needs_cleanup(dictation: str) -> bool:
    """Whether the dictation has anything the model is for."""
    return len(dictation.split()) >= _MIN_WORDS and _CUES.search(dictation) is not None


_LOGGED_OUTCOMES = {SKIPPED: "skipped", TOO_SLOW: "discarded", REJECTED: "rejected"}


def _outcome(text: str, trace: DictationTrace, outcome: str, reason: str) -> str:
    """The dictation unchanged, with why the model's answer was not used on record."""
    _logger.info("Cleanup %s: %s.", _LOGGED_OUTCOMES[outcome], reason)
    trace.cleanup, trace.cleanup_reason = outcome, reason
    return text


def rejection_reason(
    original: str, cleaned: str, level: str, transformed: bool = False,
) -> str:
    """Why the model's answer must not replace the dictation, or ``""``."""
    answer = cleaned.strip()
    if not answer:
        return "the answer was empty"
    if "<think>" in answer or "</think>" in answer:
        return "the answer contained reasoning"
    ratio = len(answer) / max(len(original.strip()), 1)
    if ratio > _MAX_LENGTH_RATIO or (ratio < _MIN_LENGTH_RATIO and not transformed):
        return f"the length changed by a factor of {ratio:.2f}"
    if _PREAMBLES.match(answer) and not _PREAMBLES.match(original.strip()):
        return "the answer talked to the speaker"
    if _is_quoted(answer) and not _is_quoted(original.strip()):
        return "the answer was wrapped in quotes"
    if not _numbers_kept(original, answer):
        return "a number changed"
    if len(_NEGATION.findall(answer)) < len(_NEGATION.findall(original)):
        return "a negation was dropped"
    if _hedge_count(answer) < _hedge_count(original):
        return "a hedge was dropped"
    if not transformed and (added := _added_words(original, answer)):
        return "words were added: " + ", ".join(sorted(added)[:5])
    if transformed:
        return ""
    if _LIST_MARKER.search(answer) and not _LIST_MARKER.search(original):
        return "a list was made"
    if "\n" in answer and "\n" not in original.strip():
        return "line breaks were added"
    return ""


def _added_words(original: str, cleaned: str) -> set[str]:
    """Words in the answer the speaker never said; at any level, the model may
    only drop words, never put words in the speaker's mouth."""
    spoken = {word.casefold() for word in _WORD.findall(original)}
    return {word.casefold() for word in _WORD.findall(cleaned)} - spoken


def _numbers_kept(original: str, cleaned: str) -> bool:
    before = Counter(_NUMBER.findall(_LIST_MARKER.sub("", original)))
    after = Counter(_NUMBER.findall(_LIST_MARKER.sub("", cleaned)))
    taken_back = Counter(match["old"] for match in _CORRECTED_NUMBER.finditer(original))
    return not (after - before) and not ((before - taken_back) - after)


def _hedge_count(text: str) -> int:
    lowered = text.casefold()
    return sum(len(re.findall(rf"\b{re.escape(hedge)}\b", lowered)) for hedge in _HEDGES)


def _is_quoted(text: str) -> bool:
    return len(text) > 1 and any(
        text.startswith(opening) and text.endswith(closing)
        for opening, closing in _QUOTE_PAIRS
    )


def _with_edges_of(original: str, cleaned: str) -> str:
    """Keep the dictation's own leading and trailing whitespace."""
    lead = original[: len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    return lead + cleaned + trail
