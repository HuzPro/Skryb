"""Telling the speech model which names and terms may come up.

Qwen3-ASR reads the list, notes included, as text before the audio, so it
can tell "Skryb (the app I'm building)" from the ordinary word. Phonon's
decoder favours up to 25 of the entries. The other engines hear nothing of it.
"""

import logging
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from skryb.people.terms import Term
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.qwen3_asr_prompt import PromptedTranscriber
from skryb.transcription.result import TranscriptionResult

_logger = logging.getLogger(__name__)

_PREAMBLE = "Names and terms that may come up"
# What the model writes when it reads the hint back instead of the speech, with or without the first words.
_ECHO = "terms that may come up"
# On shorter clips the hint makes the model invent words ("Oh." became "Boston.") or read the hint back:
# re-decoding 1,393 of Huz's dictations with it, 39 of 63 clips under 1 s came out as the hint and 4 of
# 145 at 1-1.5 s, none from 2 s up, and under 2 s it changed 3 times as many words as it fixed names.
MIN_HINT_SECONDS = 2.0
HOTWORD_LIMIT = 25


@runtime_checkable
class HotwordTranscriber(Protocol):
    """An engine whose decoder can favour a short list of words (Phonon on Fermion's CPU engine)."""

    def set_hotwords(self, words: Sequence[str]) -> None: ...


def hint_text(terms: Sequence[Term], names: Iterable[str] = ()) -> str | None:
    """The hint for one clip: every entry with its note, then any *names* (a meeting's speakers) not in the list."""
    items = [f"{term.written} ({term.note})" if term.note else term.written for term in terms]
    known = {term.written.casefold() for term in terms}
    items += [name for name in dict.fromkeys(n.strip() for n in names) if name and name.casefold() not in known]
    return f"{_PREAMBLE}: {', '.join(items)}." if items else None


def echoes_hint(text: str, hint: str) -> bool:
    """Whether a transcript is the hint read back: its opening words, or three of its entries in a row."""
    folded = text.casefold()
    if _ECHO in folded:
        return True
    items = hint.partition(": ")[2].rstrip(".").split(", ")
    return any(", ".join(items[at:at + 3]).casefold() in folded for at in range(len(items) - 2))


def transcribe_with_hint(
    engine: PromptedTranscriber, audio: AudioSegment, hint: str | None, language: str | None,
) -> TranscriptionResult:
    """Qwen3-ASR told what may come up, except on a clip too short to trust with it or when it reads the hint back."""
    if hint is None or audio.duration_seconds < MIN_HINT_SECONDS:
        return engine.transcribe_prompted(audio, None, language)
    result = engine.transcribe_prompted(audio, hint, language)
    if not echoes_hint(result.text, hint):
        return result
    _logger.warning("The transcript read the names hint back; transcribing the clip again without it.")
    return engine.transcribe_prompted(audio, None, language)


def transcribe_with_terms(
    engine: TranscriptionEngine, audio: AudioSegment, terms: Sequence[Term], language: str,
) -> TranscriptionResult:
    """One dictation, with the list handed to whichever engine can use it."""
    if isinstance(engine, PromptedTranscriber):
        return transcribe_with_hint(engine, audio, hint_text(terms), language)
    if isinstance(engine, HotwordTranscriber):
        engine.set_hotwords([term.written for term in terms[:HOTWORD_LIMIT]])
    return engine.transcribe(audio)
