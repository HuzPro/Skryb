"""How much has been dictated: the words kept in history, the time spent recording them, and what Home compares them with."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import median

from skryb.history.entries import CANCELLED, EMPTY, FAILED, NOT_SAVED, HistoryEntry

# Home's activity chart, pace, typical dictation and reliability cover this many days, today last.
CHART_DAYS = 30
# A usual day is the median of the days dictated in the four weeks before today, once there are this many.
USUAL_DAY_WINDOW = 28
USUAL_DAY_MINIMUM = 5
# A usual week is the median of up to eight earlier weeks, once there are two.
USUAL_WEEKS = 8
USUAL_WEEK_MINIMUM = 2
# Reliability means little over a handful of recordings.
RELIABILITY_MINIMUM = 10
# Recordings that took time and typed nothing; time saved pays for them.
_LOST = frozenset({CANCELLED, EMPTY, FAILED})


@dataclass(frozen=True)
class Usage:
    words: int = 0
    seconds: float = 0.0
    dictations: int = 0
    # Seconds spent on recordings that were cancelled, came back empty or failed.
    lost_seconds: float = 0.0


@dataclass(frozen=True)
class Reliability:
    """Recordings over the last 30 days that typed something, against those that came back empty or failed."""

    typed: int = 0
    empty: int = 0
    failed: int = 0

    @property
    def recorded(self) -> int:
        return self.typed + self.empty + self.failed


@dataclass(frozen=True)
class UsageReport:
    """Everything dictation Home shows, with "usual" figures to compare against; None where there is too little history."""

    all: Usage
    week: Usage
    today: Usage
    month: Usage
    streak: int
    longest_streak: int
    days: tuple[int, ...]
    usual_day: int | None
    usual_week: int | None
    typical_words: int | None
    typical_seconds: float | None
    reliability: Reliability


def usage(entries: Iterable[HistoryEntry]) -> Usage:
    """Words and recording time over the dictations that kept their text, and the time lost to ones that typed nothing.

    A dictation saved without its text (history off) has no words to count, so its time is left out of both.
    """
    words, seconds, dictations, lost = 0, 0.0, 0, 0.0
    for entry in entries:
        if entry.status in _LOST:
            lost += entry.audio_seconds
        if entry.status:
            continue
        words += len(entry.shown_text.split())
        seconds += entry.audio_seconds
        dictations += 1
    return Usage(words, seconds, dictations, lost)


def report(entries: list[HistoryEntry], today: date, day_of: Callable[[HistoryEntry], date]) -> UsageReport:
    """Home's numbers, with *day_of* giving the local day each dictation was made."""
    dated = [(day_of(entry), entry) for entry in entries]
    monday = week_start(today)
    first = today - timedelta(days=CHART_DAYS - 1)
    words_by_day: dict[date, int] = {}
    for day, entry in dated:
        if not entry.status:
            words_by_day[day] = words_by_day.get(day, 0) + len(entry.shown_text.split())
    recent = [entry for day, entry in dated if first <= day <= today]
    recent_kept = [entry for entry in recent if not entry.status]
    dictated = {day for day, entry in dated if not entry.status or entry.status == NOT_SAVED}
    return UsageReport(
        all=usage(entry for _, entry in dated),
        week=usage(entry for day, entry in dated if monday <= day <= today),
        today=usage(entry for day, entry in dated if day == today),
        month=usage(recent),
        streak=streak(dictated, today),
        longest_streak=longest_streak(dictated),
        days=tuple(words_by_day.get(first + timedelta(days=offset), 0) for offset in range(CHART_DAYS)),
        usual_day=usual_day(words_by_day, today),
        usual_week=usual_week(words_by_day, today),
        typical_words=round(median(len(entry.shown_text.split()) for entry in recent_kept)) if recent_kept else None,
        typical_seconds=round(median(entry.audio_seconds for entry in recent_kept), 1) if recent_kept else None,
        reliability=Reliability(
            typed=sum(1 for entry in recent if not entry.status or entry.status == NOT_SAVED),
            empty=sum(1 for entry in recent if entry.status == EMPTY),
            failed=sum(1 for entry in recent if entry.status == FAILED),
        ),
    )


def usual_day(words_by_day: dict[date, int], today: date) -> int | None:
    """The median words on the days dictated in the four weeks before today; None with fewer than five such days."""
    start = today - timedelta(days=USUAL_DAY_WINDOW)
    days = [words for day, words in words_by_day.items() if start <= day < today and words > 0]
    return round(median(days)) if len(days) >= USUAL_DAY_MINIMUM else None


def usual_week(words_by_day: dict[date, int], today: date) -> int | None:
    """The median words of earlier weeks from Monday to today's weekday, so a week so far meets its like.

    Only weeks since the first dictation count, so a new user's empty past doesn't drag it down.
    """
    if not words_by_day:
        return None
    began = week_start(min(words_by_day))
    monday = week_start(today)
    span = (today - monday).days
    totals = []
    for back in range(1, USUAL_WEEKS + 1):
        start = monday - timedelta(weeks=back)
        if start < began:
            break
        totals.append(sum(words_by_day.get(start + timedelta(days=offset), 0) for offset in range(span + 1)))
    return round(median(totals)) if len(totals) >= USUAL_WEEK_MINIMUM else None


def streak(days: set[date], today: date) -> int:
    """Days in a row with a dictation, back from today; not having dictated yet today doesn't break it."""
    day = today if today in days else today - timedelta(days=1)
    count = 0
    while day in days:
        count += 1
        day -= timedelta(days=1)
    return count


def longest_streak(days: set[date]) -> int:
    """The most days in a row ever dictated."""
    longest = 0
    for day in days:
        if day - timedelta(days=1) in days:
            continue
        length = 1
        while day + timedelta(days=length) in days:
            length += 1
        longest = max(longest, length)
    return longest


def week_start(today: date) -> date:
    """The Monday of *today*'s week."""
    return today - timedelta(days=today.weekday())
