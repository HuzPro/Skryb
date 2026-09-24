import json
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

from voicetotext.history.entries import (
    CANCELLED,
    EMPTY,
    FAILED,
    NOT_SAVED,
    HistoryEntry,
    empty_state_message,
    filter_entries,
    load_history,
    preview,
)

_SESSION = "20260924T063505Z-7aff44ec"
_START = datetime(2026, 9, 24, 6, 35, 5, tzinfo=UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _transcription(index: int, text: str, at: datetime, **overrides) -> dict:
    event = {
        "event": "transcription",
        "index": index,
        "requested_at": _iso(at),
        "finished_at": _iso(at + timedelta(seconds=0.3)),
        "audio_seconds": 2.368,
        "audio_path": f"audio/{index:04d}.wav",
        "text": text,
        "transcript_path": f"transcripts/{index:04d}.txt",
        "model": "parakeet-tdt-0.6b-v3",
        "engine": "nano_parakeet",
        "is_empty": not text.strip(),
        "error": None,
    }
    event.update(overrides)
    return event


def _cancelled(index: int, at: datetime, **overrides) -> dict:
    event = {
        "event": "cancelled",
        "index": index,
        "at": _iso(at),
        "audio_seconds": 4.2,
        "audio_path": f"audio/{index:04d}.wav",
        "model": "parakeet-tdt-0.6b-v3",
        "engine": "nano_parakeet",
    }
    event.update(overrides)
    return event


def _write_session(root: Path, name: str, lines: list[dict | str]) -> Path:
    session = root / name
    session.mkdir(parents=True)
    with (session / "session.jsonl").open("w", encoding="utf-8") as stream:
        for line in lines:
            if isinstance(line, dict):
                line = json.dumps(line, ensure_ascii=False)
            stream.write(line + "\n")
    return session


def _entry(text: str, **overrides) -> HistoryEntry:
    fields = {
        "key": (_SESSION, 0),
        "at": _START,
        "text": text,
        "audio_seconds": 1.0,
        "audio_path": None,
        "status": "",
        "error": None,
    }
    fields.update(overrides)
    return HistoryEntry(**fields)


def test_lists_the_newest_dictation_first(tmp_path: Path) -> None:
    _write_session(tmp_path, "20260923T080000Z-11111111", [
        _transcription(1, "yesterday morning", _START - timedelta(days=1)),
    ])
    _write_session(tmp_path, _SESSION, [
        {"event": "app_started", "started_at": _iso(_START)},
        _transcription(1, "first today", _START + timedelta(minutes=1)),
        _transcription(2, "second today", _START + timedelta(minutes=2)),
        {"event": "app_stopped", "stopped_at": _iso(_START + timedelta(hours=1))},
    ])

    texts = [entry.text for entry in load_history(tmp_path)]

    assert texts == ["second today", "first today", "yesterday morning"]


def test_lists_thousands_of_dictations_across_hundreds_of_sessions(
    tmp_path: Path,
) -> None:
    for session_number in range(200):
        started = _START - timedelta(hours=session_number)
        name = started.strftime("%Y%m%dT%H%M%SZ") + f"-{session_number:08x}"
        _write_session(tmp_path, name, [
            _transcription(i, f"session {session_number} line {i}",
                           started + timedelta(seconds=i))
            for i in range(1, 26)
        ])

    entries = load_history(tmp_path)

    assert len(entries) == 5000
    assert entries[0].text == "session 0 line 25"
    assert entries[-1].text == "session 199 line 1"
    assert all(a.at >= b.at for a, b in pairwise(entries))


def test_skips_malformed_lines_and_a_half_written_last_line(tmp_path: Path) -> None:
    good = _transcription(1, "kept", _START)
    half_written = json.dumps(_transcription(2, "lost", _START))[:40]
    _write_session(tmp_path, _SESSION, [
        "not json at all",
        "",
        "[1, 2, 3]",
        "null",
        '"just a string"',
        good,
        half_written,
    ])

    entries = load_history(tmp_path)

    assert [entry.text for entry in entries] == ["kept"]


def test_ignores_folders_and_files_that_are_not_sessions(tmp_path: Path) -> None:
    _write_session(tmp_path, _SESSION, [_transcription(1, "real", _START)])
    _write_session(tmp_path, "notes", [_transcription(1, "not a session", _START)])
    _write_session(tmp_path, "20260924T063505Z-NOTHEX00", [
        _transcription(1, "bad name", _START),
    ])
    _write_session(tmp_path, "20261399T999999Z-abcdef12", [
        _transcription(1, "impossible date", _START),
    ])
    (tmp_path / "20260925T000000Z-deadbeef").mkdir()  # no session.jsonl yet
    (tmp_path / "20260926T000000Z-cafef00d").write_text("a file, not a folder")

    assert [entry.text for entry in load_history(tmp_path)] == ["real"]


def test_returns_nothing_for_a_missing_logs_folder(tmp_path: Path) -> None:
    assert load_history(tmp_path / "does-not-exist") == []


def test_keeps_duplicate_dictations_as_separate_entries(tmp_path: Path) -> None:
    _write_session(tmp_path, _SESSION, [
        _transcription(1, "Send it over.", _START),
        _transcription(2, "Send it over.", _START),
        _transcription(2, "Send it over.", _START),
    ])

    entries = load_history(tmp_path)

    assert [entry.text for entry in entries] == ["Send it over."] * 3
    assert len({entry.key for entry in entries}) == 3


def test_reads_the_transcript_file_when_the_log_line_has_no_text(
    tmp_path: Path,
) -> None:
    session = _write_session(tmp_path, _SESSION, [
        _transcription(1, "", _START, is_empty=False),
        _transcription(2, "from the log", _START + timedelta(seconds=1)),
    ])
    (session / "transcripts").mkdir()
    (session / "transcripts" / "0001.txt").write_text(
        "from the file", encoding="utf-8",
    )
    (session / "transcripts" / "0002.txt").write_text(
        "stale file copy", encoding="utf-8",
    )

    entries = load_history(tmp_path)

    assert [entry.text for entry in entries] == ["from the log", "from the file"]
    assert entries[1].status == ""


def test_marks_a_cancelled_recording_and_keeps_its_audio(tmp_path: Path) -> None:
    session = _write_session(tmp_path, _SESSION, [_cancelled(3, _START)])

    (entry,) = load_history(tmp_path)

    assert entry.status == CANCELLED
    assert entry.is_cancelled
    assert entry.text == ""
    assert entry.audio_seconds == 4.2
    assert entry.audio_path == session / "audio" / "0003.wav"


def test_marks_failed_empty_and_unsaved_dictations(tmp_path: Path) -> None:
    _write_session(tmp_path, _SESSION, [
        _transcription(1, "", _START, audio_path="", transcript_path="",
                       error="CUDA failed with error out of memory"),
        _transcription(2, "   ", _START + timedelta(seconds=1)),
        # Content capture off: the result had words but none were kept.
        _transcription(3, "", _START + timedelta(seconds=2), audio_path="",
                       transcript_path="", is_empty=False),
        _transcription(4, "Fine.", _START + timedelta(seconds=3)),
    ])

    statuses = [entry.status for entry in load_history(tmp_path)]

    assert statuses == ["", NOT_SAVED, EMPTY, FAILED]


def test_tolerates_missing_blank_and_wrongly_typed_fields(tmp_path: Path) -> None:
    _write_session(tmp_path, _SESSION, [
        {"event": "transcription"},
        {"event": "transcription", "requested_at": "yesterday", "audio_seconds": "3",
         "audio_path": None, "text": None, "error": "   ", "is_empty": "no"},
        {"event": "transcription", "requested_at": _iso(_START), "audio_seconds": True},
        '{"event": "transcription", "audio_seconds": NaN}',
        {"event": "transcription", "audio_seconds": -2.5},
        {"event": "cancelled"},
    ])

    entries = load_history(tmp_path)

    assert len(entries) == 6
    assert {entry.at for entry in entries} == {_START}
    assert {entry.audio_seconds for entry in entries} == {0.0}
    assert {entry.audio_path for entry in entries} == {None}
    assert {entry.error for entry in entries} == {None}
    assert sorted(entry.status for entry in entries) == [CANCELLED] + [EMPTY] * 5


def test_refuses_audio_paths_that_leave_the_session_folder(tmp_path: Path) -> None:
    _write_session(tmp_path, _SESSION, [
        _transcription(1, "a", _START, audio_path="../other/audio/0001.wav"),
        _transcription(2, "b", _START, audio_path="C:/Windows/media/chimes.wav"),
        _transcription(3, "c", _START, audio_path="/etc/passwd"),
    ])

    assert {entry.audio_path for entry in load_history(tmp_path)} == {None}


def test_keeps_unicode_text_and_survives_undecodable_bytes(tmp_path: Path) -> None:
    text = "Café naïve — مرحبا بك 你好 🎙️"
    session = _write_session(tmp_path, _SESSION, [
        _transcription(1, text, _START),
        _transcription(2, "", _START + timedelta(seconds=1), is_empty=False),
    ])
    (session / "transcripts").mkdir()
    (session / "transcripts" / "0002.txt").write_bytes(b"ok \xff\xfe broken")
    with (session / "session.jsonl").open("ab") as stream:
        stream.write(b'{"event":"transcription","text":"\xff\xfe"}\n')

    entries = load_history(tmp_path)

    assert text in [entry.text for entry in entries]
    assert any(entry.text.startswith("ok ") for entry in entries)
    assert len(entries) == 3


def test_search_matches_text_ignoring_case_and_surrounding_spaces() -> None:
    entries = [
        _entry("Meet me on Hauptstraße at noon"),
        _entry("Send the invoice"),
        _entry(""),
    ]

    assert filter_entries(entries, "  HAUPTSTRASSE ") == [entries[0]]
    assert filter_entries(entries, "INVOICE") == [entries[1]]
    assert filter_entries(entries, "nothing like this") == []


def test_a_blank_search_shows_everything() -> None:
    entries = [_entry("one"), _entry("one"), _entry("")]

    assert filter_entries(entries, "   ") == entries


def test_preview_flattens_lines_and_shortens_long_text() -> None:
    assert preview(_entry("first line\n\n  second\tline ")) == "first line second line"
    long_preview = preview(_entry("word " * 100))
    assert len(long_preview) == 120
    assert long_preview.endswith("\u2026")
    assert preview(_entry("x" * 120)) == "x" * 120


def test_preview_of_a_failed_dictation_shows_the_error() -> None:
    assert preview(_entry("", error="CUDA out of memory")) == "(CUDA out of memory)"


def test_preview_of_a_cancelled_recording_says_it_was_never_transcribed() -> None:
    assert preview(_entry("", status="Cancelled")) == "(cancelled - not transcribed)"


def test_explains_an_empty_history_and_how_to_fill_it(tmp_path: Path) -> None:
    missing = empty_state_message([], tmp_path / "gone")
    empty = empty_state_message([], tmp_path)

    assert "does not exist" in missing
    assert empty.startswith("No dictations yet.")
    for message in (missing, empty):
        assert "Settings > Output" in message


def test_hints_at_content_capture_when_the_newest_dictation_was_not_saved() -> None:
    unsaved = [_entry("", status=NOT_SAVED), _entry("older, saved")]
    saved = [_entry("newest, saved"), _entry("", status=NOT_SAVED)]

    assert "not saved" in empty_state_message(unsaved, Path("."))
    assert empty_state_message(saved, Path(".")) == ""
