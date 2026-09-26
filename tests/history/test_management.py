import json
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from voicetotext.diagnostics.jsonl import JsonlWriter
from voicetotext.history.entries import HistoryEntry, load_history
from voicetotext.history.management import (
    default_export_name,
    delete_all_history,
    delete_entries,
    export_audio,
)
from voicetotext.history.retention import history_retention_days, prune_history
from tests.fakes import FakeConfig

_NOW = datetime(2026, 9, 26, 12, 0, 0, tzinfo=UTC)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _session_name(started: datetime, suffix: int) -> str:
    return f"{started:%Y%m%dT%H%M%SZ}-{suffix:08x}"


def _dictation(index: int, text: str, at: datetime, **overrides) -> dict:
    event = {
        "event": "transcription",
        "index": index,
        "requested_at": _iso(at),
        "audio_seconds": 2.4,
        "audio_path": f"audio/{index:04d}.wav",
        "text": text,
        "transcript_path": f"transcripts/{index:04d}.txt",
        "is_empty": not text.strip(),
        "error": None,
    }
    event.update(overrides)
    return event


def _cancelled(index: int, at: datetime) -> dict:
    return {
        "event": "cancelled", "index": index, "at": _iso(at),
        "audio_seconds": 4.2, "audio_path": f"audio/{index:04d}.wav",
    }


def _write_session(root: Path, name: str, events: list[dict]) -> Path:
    """A session as diagnostics writes it: log lines plus each entry's files."""
    session = root / name
    (session / "audio").mkdir(parents=True, exist_ok=True)
    (session / "transcripts").mkdir(exist_ok=True)
    (session / "session.log").write_text("started\n", encoding="utf-8")
    (session / "edits.jsonl").write_text('{"event":"delivered"}\n', encoding="utf-8")
    with (session / "session.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "app_started"}) + "\n")
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            if event.get("audio_path"):
                (session / event["audio_path"]).write_bytes(b"RIFF" + b"\x00" * 40)
            if event.get("transcript_path"):
                (session / event["transcript_path"]).write_text(
                    event.get("text", ""), encoding="utf-8",
                )
    return session


def _texts(root: Path) -> list[str]:
    return [entry.text for entry in load_history(root)]


def _entry_with_seq(root: Path, seq: int) -> HistoryEntry:
    return next(entry for entry in load_history(root) if entry.seq == seq)


def test_deleting_one_entry_keeps_its_duplicates_and_its_neighbours(tmp_path) -> None:
    start = _NOW - timedelta(hours=1)
    name = _session_name(start, 1)
    session = _write_session(tmp_path, name, [
        _dictation(1, "Send it over.", start),
        _dictation(2, "Send it over.", start + timedelta(seconds=5)),
        _dictation(3, "Send it over.", start + timedelta(seconds=9)),
        _dictation(4, "Déjà vu — ünïcödé ✓", start + timedelta(seconds=12)),
    ])

    deleted = delete_entries(tmp_path, [_entry_with_seq(tmp_path, 2)], _NOW)

    assert deleted == 1
    assert sorted(entry.seq for entry in load_history(tmp_path)) == [1, 3, 4]
    assert _texts(tmp_path).count("Send it over.") == 2
    assert not (session / "audio" / "0002.wav").exists()
    assert not (session / "transcripts" / "0002.txt").exists()
    for kept in (1, 3, 4):
        assert (session / "audio" / f"{kept:04d}.wav").exists()
        assert (session / "transcripts" / f"{kept:04d}.txt").exists()


def test_a_deletion_only_appends_to_the_session_log(tmp_path) -> None:
    start = _NOW - timedelta(hours=2)
    session = _write_session(tmp_path, _session_name(start, 2), [
        _dictation(1, "keep me", start),
        _dictation(2, "delete me", start + timedelta(seconds=1)),
    ])
    before = (session / "session.jsonl").read_text(encoding="utf-8")

    delete_entries(tmp_path, [_entry_with_seq(tmp_path, 2)], _NOW)

    after = (session / "session.jsonl").read_text(encoding="utf-8")
    assert after.startswith(before)
    tombstone = json.loads(after[len(before):])
    assert tombstone == {"event": "history_deleted", "seq": 2, "at": "2026-09-26T12:00:00Z"}


def test_tombstones_hide_entries_after_a_reload_and_deleting_again_is_harmless(
    tmp_path,
) -> None:
    start = _NOW - timedelta(days=3)
    _write_session(tmp_path, _session_name(start, 3), [
        _dictation(1, "first", start),
        _dictation(2, "second", start + timedelta(seconds=1)),
    ])
    stale = load_history(tmp_path)
    doomed = [entry for entry in stale if entry.seq == 1]

    delete_entries(tmp_path, doomed, _NOW)
    delete_entries(tmp_path, doomed, _NOW + timedelta(seconds=1))

    assert _texts(tmp_path) == ["second"]
    assert _texts(tmp_path) == ["second"]


def test_an_entry_whose_files_were_never_saved_can_still_be_deleted(tmp_path) -> None:
    start = _NOW - timedelta(minutes=30)
    _write_session(tmp_path, _session_name(start, 4), [
        _dictation(1, "", start, audio_path="", transcript_path="", is_empty=False),
        _dictation(2, "", start + timedelta(seconds=1), audio_path="",
                   error="CUDA failed with error out of memory"),
        _cancelled(3, start + timedelta(seconds=2)),
    ])
    (tmp_path / _session_name(start, 4) / "audio" / "0003.wav").unlink()

    deleted = delete_entries(tmp_path, load_history(tmp_path), _NOW)

    assert deleted == 3
    assert load_history(tmp_path) == []


def test_a_tombstone_for_a_seq_that_matches_nothing_hides_nothing(tmp_path) -> None:
    start = _NOW - timedelta(minutes=5)
    session = _write_session(tmp_path, _session_name(start, 5), [
        _dictation(1, "still here", start),
    ])
    JsonlWriter(session / "session.jsonl").append_many([
        {"event": "history_deleted", "seq": 99, "at": _iso(_NOW)},
        {"event": "history_deleted", "seq": True, "at": _iso(_NOW)},
        {"event": "history_deleted", "seq": "1", "at": _iso(_NOW)},
    ])

    assert _texts(tmp_path) == ["still here"]


def test_files_the_log_points_outside_the_entry_folders_are_never_touched(
    tmp_path,
) -> None:
    start = _NOW - timedelta(hours=4)
    victim = _write_session(tmp_path, _session_name(start, 6), [
        _dictation(1, "someone else's recording", start),
    ])
    outside = tmp_path / "precious.wav"
    outside.write_bytes(b"RIFF")
    session = _write_session(tmp_path, _session_name(start, 7), [])
    with (session / "session.jsonl").open("a", encoding="utf-8") as stream:
        for index, (audio, transcript) in enumerate([
            (f"../{victim.name}/audio/0001.wav", f"../{victim.name}/transcripts/0001.txt"),
            (str(outside), str(outside)),
            ("session.jsonl", "session.log"),
            ("edits.jsonl", "audio"),
        ], start=1):
            stream.write(json.dumps(_dictation(
                index, f"hostile line {index}", start,
                audio_path=audio, transcript_path=transcript,
            )) + "\n")
    hostile = [entry for entry in load_history(tmp_path) if entry.key[0] == session.name]
    log_before = (session / "session.jsonl").read_text(encoding="utf-8")

    delete_entries(tmp_path, hostile, _NOW)

    assert (victim / "audio" / "0001.wav").exists()
    assert (victim / "transcripts" / "0001.txt").exists()
    assert outside.exists()
    assert (session / "session.log").read_text(encoding="utf-8") == "started\n"
    assert (session / "edits.jsonl").exists()
    assert (session / "audio").is_dir()
    assert (session / "session.jsonl").read_text(encoding="utf-8").startswith(log_before)
    assert _texts(tmp_path) == ["someone else's recording"]


def test_delete_all_empties_history_across_thousands_of_entries_and_leaves_logs_alone(
    tmp_path,
) -> None:
    logs: dict[Path, tuple[str, str]] = {}
    for session_number in range(50):
        start = _NOW - timedelta(days=session_number, hours=1)
        events: list[dict] = []
        for index in range(1, 51):
            at = start + timedelta(seconds=index)
            if index % 10 == 0:
                events.append(_cancelled(index, at))
            elif index % 7 == 0:
                events.append(_dictation(index, "", at, error="engine crashed"))
            elif index % 5 == 0:
                events.append(_dictation(index, "   ", at))
            else:
                events.append(_dictation(index, f"Dictation {index} — naïve café 東京", at))
        session = _write_session(tmp_path, _session_name(start, session_number), events)
        logs[session] = (
            (session / "session.log").read_text(encoding="utf-8"),
            (session / "edits.jsonl").read_text(encoding="utf-8"),
        )
    jsonl_before = {
        session: (session / "session.jsonl").read_text(encoding="utf-8")
        for session in logs
    }
    assert len(load_history(tmp_path)) == 2500

    deleted = delete_all_history(tmp_path, _NOW)

    assert deleted == 2500
    assert load_history(tmp_path) == []
    for session, (session_log, edits) in logs.items():
        assert (session / "session.log").read_text(encoding="utf-8") == session_log
        assert (session / "edits.jsonl").read_text(encoding="utf-8") == edits
        assert (session / "session.jsonl").read_text(encoding="utf-8").startswith(
            jsonl_before[session],
        )
        assert not any((session / "audio").iterdir())
        assert not any((session / "transcripts").iterdir())


def test_deleting_from_the_session_being_written_never_corrupts_its_log(
    tmp_path,
) -> None:
    start = _NOW - timedelta(minutes=10)
    session = _write_session(tmp_path, _session_name(start, 8), [
        _dictation(index, f"line {index}", start + timedelta(seconds=index))
        for index in range(1, 301)
    ])
    diagnostics = JsonlWriter(session / "session.jsonl")
    entries = load_history(tmp_path)

    def keep_logging() -> None:
        for count in range(2000):
            diagnostics.append({"event": "engine_loaded", "count": count, "text": "x" * 200})

    writer = threading.Thread(target=keep_logging)
    writer.start()
    for entry in entries:
        delete_entries(tmp_path, [entry], _NOW)
    writer.join()

    lines = (session / "session.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert sum(event["event"] == "engine_loaded" for event in events) == 2000
    assert sum(event["event"] == "history_deleted" for event in events) == 300
    assert load_history(tmp_path) == []


def test_export_copies_the_recording_and_leaves_history_as_it_was(tmp_path) -> None:
    start = datetime(2026, 9, 25, 8, 30, 15, tzinfo=UTC)
    session = _write_session(tmp_path, _session_name(start, 9), [
        _dictation(1, "export me", start),
    ])
    (session / "audio" / "0001.wav").write_bytes(b"RIFF" + bytes(range(256)) * 4)
    entry = load_history(tmp_path)[0]
    log_before = (session / "session.jsonl").read_text(encoding="utf-8")
    destination = tmp_path / "exports" / default_export_name(entry)
    destination.parent.mkdir()

    export_audio(tmp_path, entry, destination)

    assert destination.read_bytes() == (session / "audio" / "0001.wav").read_bytes()
    assert (session / "session.jsonl").read_text(encoding="utf-8") == log_before
    assert load_history(tmp_path) == [entry]
    assert destination.name.startswith("dictation-2026") and destination.suffix == ".wav"


def test_export_refuses_a_recording_outside_its_session_folder(tmp_path) -> None:
    start = _NOW - timedelta(hours=1)
    session = _write_session(tmp_path, _session_name(start, 10), [
        _dictation(1, "legit", start),
    ])
    elsewhere = tmp_path / "elsewhere.wav"
    elsewhere.write_bytes(b"RIFF")
    entry = load_history(tmp_path)[0]
    forged = HistoryEntry(
        key=entry.key, at=entry.at, text=entry.text,
        audio_seconds=entry.audio_seconds, audio_path=session / ".." / "elsewhere.wav",
        status=entry.status, error=None, seq=entry.seq,
    )
    destination = tmp_path / "copy.wav"

    with pytest.raises(ValueError):
        export_audio(tmp_path, forged, destination)

    assert not destination.exists()


def test_retention_deletes_only_entries_older_than_the_boundary(tmp_path) -> None:
    boundary = _NOW - timedelta(days=30)
    old_start = boundary - timedelta(days=2)
    _write_session(tmp_path, _session_name(old_start, 11), [
        _dictation(1, "two days past the limit", old_start),
        _dictation(2, "a microsecond past the limit", boundary - timedelta(microseconds=1)),
        _dictation(3, "exactly at the limit", boundary),
        _dictation(4, "a second inside the limit", boundary + timedelta(seconds=1)),
    ])
    _write_session(tmp_path, _session_name(_NOW - timedelta(hours=1), 12), [
        _dictation(1, "today", _NOW - timedelta(hours=1)),
    ])

    pruned = prune_history(tmp_path, 30, _NOW)

    assert pruned == 2
    assert sorted(_texts(tmp_path)) == [
        "a second inside the limit", "exactly at the limit", "today",
    ]


def test_keeping_history_forever_prunes_nothing(tmp_path) -> None:
    ancient = _NOW - timedelta(days=4000)
    _write_session(tmp_path, _session_name(ancient, 13), [
        _dictation(1, "from years ago", ancient),
    ])

    assert prune_history(tmp_path, 0, _NOW) == 0
    assert _texts(tmp_path) == ["from years ago"]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, 0), (0, 0), (7, 7), (90, 90), (-5, 0), (True, 0), ("30", 0), (30.0, 0)],
)
def test_an_unreadable_retention_setting_keeps_history_forever(configured, expected) -> None:
    config = FakeConfig()
    if configured is not None:
        config.set("history.retention_days", configured)

    assert history_retention_days(config) == expected


def test_startup_pruning_waits_for_the_engine_and_runs_once(monkeypatch, tmp_path) -> None:
    from voicetotext.core.event_bus import SimpleEventBus
    from voicetotext.core.state_machine import AppState, AppStateMachine
    from voicetotext.history import retention

    runs: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        retention, "prune_in_background", lambda folder, days: runs.append((folder, days)),
    )
    bus = SimpleEventBus()
    machine = AppStateMachine(bus, initial_state=AppState.LOADING)

    retention.prune_once_ready(bus, tmp_path, lambda: 7)
    assert runs == []
    machine.transition_to(AppState.IDLE)
    machine.transition_to(AppState.RECORDING)
    machine.transition_to(AppState.IDLE)

    assert runs == [(tmp_path, 7)]
