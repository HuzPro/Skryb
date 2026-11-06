import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from skryb.history.entries import CANCELLED, EMPTY, FAILED, NOT_SAVED, HistoryEntry
from skryb.history.service import HistoryService
from skryb.history.usage import CHART_DAYS, Usage, report, streak, usage, week_start

# A Wednesday at noon: the same local date in every time zone's week.
_NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)


def _entry(text: str, seconds: float = 3.0, status: str = "", typed: str = "", at: datetime = _NOW) -> HistoryEntry:
    return HistoryEntry(
        key=("20261104T080000Z-0a1b2c3d", 0), at=at, text=text, audio_seconds=seconds,
        audio_path=None, status=status, error=None, typed=typed,
    )


def _never(_wav: Path, _on_done: object) -> None:
    raise AssertionError("usage never transcribes")


def test_an_empty_history_has_no_words_and_no_speaking_time() -> None:
    assert usage([]) == Usage(0, 0.0)


def test_counts_the_words_typed_and_the_time_spent_saying_them_across_thousands_of_dictations() -> None:
    entries = [_entry(f"Notes for the Zürich office, part {index}", seconds=2.5) for index in range(3000)]

    assert usage(entries) == Usage(words=3000 * 7, seconds=3000 * 2.5, dictations=3000)


def test_counts_what_was_typed_over_what_was_heard_and_unicode_words_split_on_spaces() -> None:
    entries = [
        _entry("um send the café menu", typed="Send the café menu."),
        _entry("東京 で 会いましょう"),
        _entry("  spaced   out   words  "),
    ]

    assert usage(entries).words == 4 + 3 + 3


def test_leaves_out_cancelled_failed_empty_and_unsaved_recordings_with_their_time() -> None:
    entries = [
        _entry("Kept and counted.", seconds=4.0),
        _entry("", seconds=9.0, status=CANCELLED),
        _entry("", seconds=9.0, status=FAILED),
        _entry("", seconds=9.0, status=EMPTY),
        _entry("", seconds=9.0, status=NOT_SAVED),
    ]

    assert usage(entries) == Usage(words=3, seconds=4.0, dictations=1)


def test_a_week_starts_on_monday() -> None:
    assert week_start(date(2026, 11, 4)) == date(2026, 11, 2)
    assert week_start(date(2026, 11, 2)) == date(2026, 11, 2)
    assert week_start(date(2026, 11, 8)) == date(2026, 11, 2)
    assert week_start(date(2027, 1, 1)) == date(2026, 12, 28)


def test_this_weeks_usage_counts_only_dictations_since_monday(tmp_path: Path) -> None:
    session = tmp_path / "20261020T080000Z-0a1b2c3d"
    session.mkdir()
    lines = [
        {"event": "transcription", "index": index, "requested_at": at.isoformat().replace("+00:00", "Z"),
         "audio_seconds": 2.0, "text": text, "is_empty": False, "error": None}
        for index, (at, text) in enumerate([
            (_NOW - timedelta(days=1), "Yesterday's reminder about the garden"),
            (_NOW - timedelta(days=9), "Last week's note"),
            (_NOW - timedelta(days=16), "An older line"),
        ])
    ]
    (session / "session.jsonl").write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")

    numbers = HistoryService(tmp_path, _never, now=lambda: _NOW).usage()

    assert numbers.all == Usage(words=5 + 3 + 3, seconds=6.0, dictations=3)
    assert numbers.week == Usage(words=5, seconds=2.0, dictations=1)
    assert numbers.today == Usage()
    assert numbers.streak == 1


def _day_of(entry: HistoryEntry) -> date:
    return entry.at.date()


def test_today_counts_only_dictations_made_today_and_leaves_out_cancelled_ones() -> None:
    today = _NOW.date()
    entries = [
        _entry("Order more oat milk", seconds=2.0),
        _entry("Call the plumber at nine", seconds=3.0),
        _entry("", seconds=8.0, status=CANCELLED),
        _entry("Yesterday's line", at=_NOW - timedelta(days=1)),
    ]

    assert report(entries, today, _day_of).today == Usage(words=4 + 5, seconds=5.0, dictations=2)


def test_the_chart_has_thirty_days_of_words_oldest_first_and_ignores_older_ones() -> None:
    today = _NOW.date()
    entries = [
        _entry("one two three"),
        _entry("four five", at=_NOW - timedelta(days=1)),
        _entry("six", at=_NOW - timedelta(days=1)),
        _entry("the oldest day on the chart", at=_NOW - timedelta(days=CHART_DAYS - 1)),
        _entry("too old to show", at=_NOW - timedelta(days=CHART_DAYS)),
    ]

    days = report(entries, today, _day_of).days

    assert len(days) == CHART_DAYS
    assert days[-1] == 3
    assert days[-2] == 3
    assert days[0] == 6
    assert sum(days) == 12


def test_a_streak_counts_the_days_in_a_row_back_from_today() -> None:
    today = date(2026, 11, 4)
    days = {today - timedelta(days=offset) for offset in range(400)}

    assert streak(days, today) == 400


def test_a_streak_survives_until_the_end_of_a_day_not_yet_dictated_in() -> None:
    today = date(2026, 11, 4)

    assert streak({date(2026, 11, 3), date(2026, 11, 2)}, today) == 2


def test_a_gap_ends_the_streak_and_no_dictations_mean_no_streak() -> None:
    today = date(2026, 11, 4)

    assert streak({today, date(2026, 11, 2), date(2026, 11, 1)}, today) == 1
    assert streak({date(2026, 11, 2)}, today) == 0
    assert streak(set(), today) == 0
