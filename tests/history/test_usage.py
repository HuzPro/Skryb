import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from skryb.history.entries import CANCELLED, EMPTY, FAILED, NOT_SAVED, HistoryEntry
from skryb.history.service import HistoryService
from skryb.history.usage import Usage, usage, week_start

# A Wednesday at noon: the same local date in every time zone's week.
_NOW = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)


def _entry(text: str, seconds: float = 3.0, status: str = "", typed: str = "") -> HistoryEntry:
    return HistoryEntry(
        key=("20261104T080000Z-0a1b2c3d", 0), at=_NOW, text=text, audio_seconds=seconds,
        audio_path=None, status=status, error=None, typed=typed,
    )


def _never(_wav: Path, _on_done: object) -> None:
    raise AssertionError("usage never transcribes")


def test_an_empty_history_has_no_words_and_no_speaking_time() -> None:
    assert usage([]) == Usage(0, 0.0)


def test_counts_the_words_typed_and_the_time_spent_saying_them_across_thousands_of_dictations() -> None:
    entries = [_entry(f"Notes for the Zürich office, part {index}", seconds=2.5) for index in range(3000)]

    assert usage(entries) == Usage(words=3000 * 7, seconds=3000 * 2.5)


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

    assert usage(entries) == Usage(words=3, seconds=4.0)


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

    everything, this_week = HistoryService(tmp_path, _never, now=lambda: _NOW).usage()

    assert everything == Usage(words=5 + 3 + 3, seconds=6.0)
    assert this_week == Usage(words=5, seconds=2.0)
