"""How much has been dictated: the words kept in history, and the time spent saying them."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta

from skryb.history.entries import HistoryEntry


# Home's activity chart covers this many days, today last.
CHART_DAYS = 30


@dataclass(frozen=True)
class Usage:
    words: int = 0
    seconds: float = 0.0
    dictations: int = 0


@dataclass(frozen=True)
class UsageReport:
    """Everything Home shows: all of history, this week, today, the streak and words a day for the chart."""

    all: Usage
    week: Usage
    today: Usage
    streak: int
    days: tuple[int, ...]


def usage(entries: Iterable[HistoryEntry]) -> Usage:
    """Words and speaking time over the dictations that kept their text.

    Cancelled, failed and empty recordings, and those saved without their
    text, have no words to count, so their speaking time is left out too.
    """
    words, seconds, dictations = 0, 0.0, 0
    for entry in entries:
        if entry.status:
            continue
        words += len(entry.shown_text.split())
        seconds += entry.audio_seconds
        dictations += 1
    return Usage(words, seconds, dictations)


def report(entries: list[HistoryEntry], today: date, day_of: Callable[[HistoryEntry], date]) -> UsageReport:
    """Home's numbers, with *day_of* giving the local day each dictation was made."""
    kept = [(day_of(entry), entry) for entry in entries if not entry.status]
    monday = week_start(today)
    first = today - timedelta(days=CHART_DAYS - 1)
    words_by_day: dict[date, int] = {}
    for day, entry in kept:
        if day >= first:
            words_by_day[day] = words_by_day.get(day, 0) + len(entry.shown_text.split())
    return UsageReport(
        all=usage(entry for _, entry in kept),
        week=usage(entry for day, entry in kept if day >= monday),
        today=usage(entry for day, entry in kept if day == today),
        streak=streak({day for day, _ in kept}, today),
        days=tuple(words_by_day.get(first + timedelta(days=offset), 0) for offset in range(CHART_DAYS)),
    )


def streak(days: set[date], today: date) -> int:
    """Days in a row with a dictation, back from today; not having dictated yet today doesn't break it."""
    day = today if today in days else today - timedelta(days=1)
    count = 0
    while day in days:
        count += 1
        day -= timedelta(days=1)
    return count


def week_start(today: date) -> date:
    """The Monday of *today*'s week."""
    return today - timedelta(days=today.weekday())
