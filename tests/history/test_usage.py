import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from skryb.history.entries import CANCELLED, EMPTY, FAILED, NOT_SAVED, HistoryEntry
from skryb.history.service import HistoryService
from skryb.history.usage import CHART_DAYS, Reliability, Usage, longest_streak, report, streak, usage, usual_day, usual_week, week_start

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


def test_counts_no_words_from_cancelled_failed_empty_or_unsaved_recordings_and_their_lost_time_separately() -> None:
    entries = [
        _entry("Kept and counted.", seconds=4.0),
        _entry("", seconds=9.0, status=CANCELLED),
        _entry("", seconds=9.0, status=FAILED),
        _entry("", seconds=9.0, status=EMPTY),
        _entry("", seconds=9.0, status=NOT_SAVED),
    ]

    assert usage(entries) == Usage(words=3, seconds=4.0, dictations=1, lost_seconds=27.0)


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

    assert report(entries, today, _day_of).today == Usage(words=4 + 5, seconds=5.0, dictations=2, lost_seconds=8.0)


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


def _words_on(*days_and_words: tuple[int, int]) -> dict[date, int]:
    today = date(2026, 11, 4)
    return {today - timedelta(days=back): words for back, words in days_and_words}


def test_a_usual_day_is_the_median_of_days_dictated_in_the_four_weeks_before_today() -> None:
    today = date(2026, 11, 4)
    words = _words_on((0, 99_999), (1, 100), (2, 300), (3, 200), (5, 5000), (9, 250), (28, 400), (29, 9_999))

    assert usual_day(words, today) == 275


def test_there_is_no_usual_day_with_fewer_than_five_days_dictated() -> None:
    today = date(2026, 11, 4)

    assert usual_day(_words_on((1, 100), (2, 300), (3, 200), (4, 50)), today) is None
    assert usual_day({}, today) is None


def test_a_usual_week_compares_monday_to_the_same_weekday_of_earlier_weeks() -> None:
    today = date(2026, 11, 4)  # A Wednesday.
    words = {
        date(2026, 10, 26): 100, date(2026, 10, 28): 50, date(2026, 10, 30): 9_000,  # Mon, Wed, then Fri.
        date(2026, 10, 19): 300,
        date(2026, 10, 12): 10, date(2026, 10, 15): 9_000,  # Thursday is past this week's Wednesday.
    }

    assert usual_week(words, today) == 150


def test_a_usual_week_counts_only_weeks_since_the_first_dictation_and_needs_two() -> None:
    today = date(2026, 11, 4)

    assert usual_week({date(2026, 10, 27): 400}, today) is None
    assert usual_week({date(2026, 10, 27): 400, date(2026, 10, 20): 0, date(2026, 10, 19): 200}, today) == 300
    assert usual_week({}, today) is None


def test_the_longest_streak_finds_the_longest_run_among_gaps() -> None:
    start = date(2026, 1, 1)
    days = {start + timedelta(days=offset) for offset in [*range(0, 3), *range(5, 45), *range(50, 51)]}

    assert longest_streak(days) == 40
    assert longest_streak(set()) == 0


def test_dictations_saved_without_text_keep_the_streak_but_add_no_words() -> None:
    today = _NOW.date()
    entries = [
        _entry("", status=NOT_SAVED),
        _entry("", status=NOT_SAVED, at=_NOW - timedelta(days=1)),
        _entry("Written down", at=_NOW - timedelta(days=2)),
    ]

    numbers = report(entries, today, _day_of)

    assert numbers.streak == 3
    assert numbers.today.words == 0


def test_typical_dictation_is_the_median_of_the_last_thirty_days_not_the_mean() -> None:
    today = _NOW.date()
    entries = [
        _entry("one two", seconds=2.0),
        _entry("one two three", seconds=3.0),
        _entry(" ".join(["long"] * 400), seconds=240.0),
        _entry("far too old to count here at all", seconds=50.0, at=_NOW - timedelta(days=CHART_DAYS)),
    ]

    numbers = report(entries, today, _day_of)

    assert numbers.typical_words == 3
    assert numbers.typical_seconds == 3.0
    assert report([], today, _day_of).typical_words is None


def test_reliability_counts_what_typed_against_empty_and_failed_and_leaves_cancelled_out() -> None:
    today = _NOW.date()
    entries = [
        *[_entry("Typed fine") for _ in range(40)],
        _entry("", status=NOT_SAVED),
        _entry("", status=EMPTY),
        _entry("", status=EMPTY),
        _entry("", status=FAILED),
        _entry("", status=CANCELLED),
        _entry("", status=FAILED, at=_NOW - timedelta(days=CHART_DAYS)),
    ]

    reliability = report(entries, today, _day_of).reliability

    assert reliability == Reliability(typed=41, empty=2, failed=1)
    assert reliability.recorded == 44
