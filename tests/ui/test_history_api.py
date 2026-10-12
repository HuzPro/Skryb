"""History over the API, against 2,500 dictations written the way the diagnostics logs write them."""

import time
import urllib.parse
import wave
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.ui.rig import Rig
from voicetotext.diagnostics.jsonl import JsonlWriter
from voicetotext.history.service import RetryOutcome

_SESSIONS = 5
_PER_SESSION = 500
_START = datetime(2026, 9, 20, 8, 0, 0, tzinfo=UTC)
_UNICODE = "Café with Aïsha at the 東京 office, number {index}"


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x10\x00" * 1600)


def _event(index: int, at: datetime, session: Path, with_audio: bool) -> dict[str, Any]:
    audio = f"audio/{index:04d}.wav"
    if with_audio:
        _write_wav(session / audio)
    if index % 17 == 0:
        return {"event": "cancelled", "index": index, "at": _iso(at), "audio_seconds": 1.5,
                "audio_path": audio if with_audio else ""}
    text = (
        "Send it over." if index % 7 == 0
        else "" if index % 11 == 0
        else _UNICODE.format(index=index) if index % 5 == 0
        else f"Notes on the quarterly report, part {index}"
    )
    return {
        "event": "transcription",
        "kind": "edit" if index % 19 == 0 else "dictation",
        "index": index,
        "requested_at": _iso(at),
        "audio_seconds": 2.4,
        "audio_path": audio if with_audio else "",
        "text": text,
        "transcript_path": "",
        "is_empty": not text.strip(),
        "error": "CUDA out of memory" if index % 13 == 0 else None,
    }


def _seed(root: Path) -> None:
    for number in range(_SESSIONS):
        started = _START + timedelta(hours=number)
        session = root / f"{started:%Y%m%dT%H%M%SZ}-{number:08x}"
        session.mkdir(parents=True)
        log = JsonlWriter(session / "session.jsonl")
        log.append({"event": "app_started", "model": "crisperwhisper-2.0-turbo"})
        for index in range(1, _PER_SESSION + 1):
            with_audio = number == _SESSIONS - 1 and index > _PER_SESSION - 12
            log.append(_event(index, started + timedelta(seconds=index), session, with_audio))


def _history(rig: Rig, query: str = "") -> list[dict[str, Any]]:
    path = "/api/history" + (f"?query={urllib.parse.quote(query)}" if query else "")
    status, entries = rig.call("GET", path)
    assert status == 200
    return entries


@pytest.fixture()
def seeded(make_rig: Callable[..., Rig]) -> Rig:
    def retranscribe(wav: Path, on_done: Callable[[RetryOutcome], None]) -> None:
        if wav.name.endswith("0500.wav"):
            on_done(RuntimeError("The engine is still loading."))
        else:
            on_done(f"Fresh text for {wav.name} — café ✓")

    rig = make_rig(retranscribe=retranscribe)
    _seed(rig.history_dir)
    return rig


def test_thousands_of_dictations_list_newest_first_and_filter_ignoring_case(seeded: Rig) -> None:
    started = time.monotonic()
    entries = _history(seeded)
    listed_in = time.monotonic() - started

    assert len(entries) == _SESSIONS * _PER_SESSION
    assert listed_in < 5.0
    assert entries == sorted(entries, key=lambda entry: entry["at"], reverse=True)
    assert set(entries[0]) == {"id", "at", "text", "audio_seconds", "status", "error", "has_audio", "kind", "writing"}
    assert datetime.fromisoformat(entries[0]["at"]).utcoffset() is not None
    assert {entry["status"] for entry in entries} == {"", "Cancelled", "Failed", "Empty"}
    assert {entry["kind"] for entry in entries} == {"dictation", "edit"}
    assert sum(entry["has_audio"] for entry in entries) == 12
    failed = next(entry for entry in entries if entry["status"] == "Failed")
    assert failed["error"] == "CUDA out of memory"

    cafe = _history(seeded, "CAFÉ WITH AÏSHA")
    unicode_count = sum(1 for entry in entries if entry["text"].startswith("Café"))
    assert len(cafe) == unicode_count > 0
    duplicates = sum(1 for entry in entries if entry["text"] == "Send it over.")
    assert len(_history(seeded, "send it OVER.")) == duplicates
    assert _history(seeded, "no dictation says this") == []


def test_a_limit_lists_only_that_many_of_the_newest_dictations(seeded: Rig) -> None:
    everything = _history(seeded)

    assert seeded.call("GET", "/api/history?limit=100") == (200, everything[:100])
    assert seeded.call("GET", "/api/history?limit=0") == (200, [])
    assert seeded.call("GET", "/api/history?limit=100000") == (200, everything)


def test_a_search_is_limited_the_same_way(seeded: Rig) -> None:
    cafe = _history(seeded, "café")

    status, first = seeded.call("GET", "/api/history?query=caf%C3%A9&limit=7")

    assert status == 200
    assert first == cafe[:7]


def test_a_limited_list_checks_the_disk_only_for_the_recordings_it_lists(
    seeded: Rig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On a spinning disk each check is a seek: 5,500 of them took 14 s.
    checked: list[Path] = []
    is_file = Path.is_file

    def counting_is_file(path: Path) -> bool:
        checked.append(path)
        return is_file(path)

    monkeypatch.setattr(Path, "is_file", counting_is_file)

    seeded.call("GET", "/api/history?limit=5")

    assert len([path for path in checked if path.suffix == ".wav"]) == 5


@pytest.mark.parametrize("limit", ["-1", "ten", "1.5", "²", "1e3"])
def test_a_limit_that_is_not_a_whole_number_is_refused(seeded: Rig, limit: str) -> None:
    status, body = seeded.call("GET", f"/api/history?limit={urllib.parse.quote(limit)}")

    assert status == 400
    assert body["error"] == "limit must be a whole number."


def test_one_dictation_can_be_exported_played_retried_and_deleted(seeded: Rig, tmp_path: Path) -> None:
    stream = seeded.events()
    entry = next(entry for entry in _history(seeded) if entry["has_audio"] and entry["id"].endswith(".499"))
    session, line = entry["id"].rsplit(".", 1)
    wav = seeded.history_dir / session / "audio" / "0499.wav"
    assert wav.is_file()

    destination = tmp_path / "Dictée exportée.wav"
    assert seeded.call("POST", f"/api/history/{entry['id']}/export", {"path": str(destination)})[0] == 204
    assert destination.read_bytes() == wav.read_bytes()

    assert seeded.call("POST", f"/api/history/{entry['id']}/play")[0] == 204
    assert seeded.call("POST", f"/api/history/{entry['id']}/stop")[0] == 204
    assert seeded.played == [wav]
    assert seeded.stopped == 1

    assert seeded.call("POST", f"/api/history/{entry['id']}/retry")[0] == 202
    assert stream.until("history") == {
        "event": "retry_done", "id": entry["id"], "text": "Fresh text for 0499.wav — café ✓",
    }

    assert seeded.call("DELETE", f"/api/history/{entry['id']}")[0] == 204
    assert stream.until("history") == {"event": "changed"}
    stream.close()
    remaining = _history(seeded)
    assert len(remaining) == _SESSIONS * _PER_SESSION - 1
    assert entry["id"] not in {kept["id"] for kept in remaining}
    assert not wav.exists()
    assert line == "499"


def test_a_failed_retry_is_reported_and_an_entry_without_audio_cannot_be_played(seeded: Rig) -> None:
    stream = seeded.events()
    entries = _history(seeded)
    last = next(entry for entry in entries if entry["has_audio"] and entry["id"].endswith(".500"))
    silent = next(entry for entry in entries if not entry["has_audio"])

    assert seeded.call("POST", f"/api/history/{last['id']}/retry")[0] == 202
    assert stream.until("history") == {
        "event": "retry_failed", "id": last["id"], "error": "The engine is still loading.",
    }
    stream.close()
    assert seeded.call("POST", f"/api/history/{silent['id']}/play") == (
        404, {"error": "No recording was saved for this entry."},
    )
    assert seeded.call("POST", f"/api/history/{silent['id']}/retry")[0] == 404
    assert seeded.played == []


def test_a_recording_that_cannot_be_played_asks_for_speakers_instead_of_a_bare_error(
    make_rig: Callable[..., Rig],
) -> None:
    def no_speakers(_wav: Path) -> None:
        raise OSError("Error querying device -1")

    rig = make_rig(play=no_speakers)
    _seed(rig.history_dir)
    playable = next(entry for entry in _history(rig) if entry["has_audio"])

    status, body = rig.call("POST", f"/api/history/{playable['id']}/play")

    assert status == 500
    assert body == {
        "error": "The recording could not be played. Check that speakers or headphones are connected.",
        "detail": "Error querying device -1",
    }


@pytest.mark.parametrize(
    "entry_id",
    ["nonsense", "..%2F..%2Fsecrets.1", "20260920T080000Z-00000000.999999", "20260920T080000Z-00000000.x"],
)
def test_an_id_that_names_no_dictation_is_not_found(seeded: Rig, entry_id: str) -> None:
    assert seeded.call("DELETE", f"/api/history/{entry_id}")[0] == 404
    assert seeded.call("POST", f"/api/history/{entry_id}/play")[0] == 404


def test_deleting_all_history_answers_at_once_and_reports_when_done(seeded: Rig) -> None:
    stream = seeded.events()

    assert seeded.call("DELETE", "/api/history")[0] == 202
    assert stream.until("history") == {"event": "deleted_all"}
    stream.close()

    assert _history(seeded) == []
    assert seeded.stopped == 1


def test_history_exports_to_a_spreadsheet_file_and_a_search_narrows_it(seeded: Rig, tmp_path: Path) -> None:
    everything = tmp_path / "Historique complet.csv"
    searched = tmp_path / "café only.csv"

    status, body = seeded.call("POST", "/api/history/export", {"path": str(everything)})
    status_searched, body_searched = seeded.call(
        "POST", f"/api/history/export?query={urllib.parse.quote('CAFÉ WITH')}", {"path": str(searched)},
    )

    assert status == status_searched == 200
    assert body["rows"] == len(_history(seeded))
    assert body_searched["rows"] == len(_history(seeded, "café with")) > 0
    assert everything.read_text(encoding="utf-8-sig").count("\n") > body["rows"]


def test_exporting_to_a_folder_that_does_not_exist_says_so(seeded: Rig, tmp_path: Path) -> None:
    status, body = seeded.call("POST", "/api/history/export", {"path": str(tmp_path / "missing" / "h.csv")})

    assert status == 400
    assert "Could not save the history" in str(body)


def test_the_writing_track_record_counts_what_the_model_did_with_recent_dictations(
    make_rig: Callable[..., Rig],
) -> None:
    rig = make_rig()
    session = rig.history_dir / f"{_START:%Y%m%dT%H%M%SZ}-0000abcd"
    session.mkdir(parents=True)
    log = JsonlWriter(session / "session.jsonl")
    outcomes = ["applied"] * 6 + ["rejected"] * 3 + ["too_slow", ""]
    for index, outcome in enumerate(outcomes, start=1):
        log.append({
            "event": "transcription", "index": index, "requested_at": _iso(_START + timedelta(seconds=index)),
            "text": f"dictation {index}", "is_empty": False, "error": None,
            "cleanup": outcome, "cleanup_reason": "a number changed" if outcome == "rejected" else "",
        })

    status, record = rig.call("GET", "/api/writing/track-record")

    assert status == 200
    assert record == {
        "dictations": 10,
        "outcomes": {"applied": 6, "rejected": 3, "too_slow": 1},
        "rejections": [{"reason": "a number changed", "count": 3}],
    }
