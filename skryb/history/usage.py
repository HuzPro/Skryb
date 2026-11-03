"""How much has been dictated: the words kept in history, and the time spent saying them."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from skryb.history.entries import HistoryEntry


@dataclass(frozen=True)
class Usage:
    words: int = 0
    seconds: float = 0.0


def usage(entries: Iterable[HistoryEntry]) -> Usage:
    """Words and speaking time over the dictations that kept their text.

    Cancelled, failed and empty recordings, and those saved without their
    text, have no words to count, so their speaking time is left out too.
    """
    words, seconds = 0, 0.0
    for entry in entries:
        if entry.status:
            continue
        words += len(entry.shown_text.split())
        seconds += entry.audio_seconds
    return Usage(words, seconds)


def week_start(today: date) -> date:
    """The Monday of *today*'s week."""
    return today - timedelta(days=today.weekday())
