"""History keeps what was typed next to what the engine heard."""

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from tests.fakes.config import FakeConfig
from tests.fakes.hardware import FakeHardwareDetector
from skryb.core.paths import AppPaths
from skryb.diagnostics.session import SessionDiagnostics
from skryb.history.entries import filter_entries, load_history, preview
from skryb.history.management import delete_entries, export_csv
from skryb.output.dictation_trace import TYPED_FIELD, DictationTrace
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine_factory import build_engine_registry
from skryb.transcription.engine_types import HardwareProfile
from skryb.transcription.result import TranscriptionResult

_SESSION = "20260928T134506Z-91220cb9"
_START = datetime(2026, 9, 28, 13, 45, 6, tzinfo=UTC)
_HARDWARE = FakeHardwareDetector(HardwareProfile(kind="cpu", device_name="Test CPU", gfx_arch=""))
_REGISTRY = build_engine_registry()
# What a dictation logged while the removed cleanup model was in use.
_OLD_CLEANUP_FIELDS = {
    "kind": "dictation",
    "cleanup": "applied",
    "cleanup_reason": "",
    "cleanup_seconds": 0.42,
    "cleanup_answer": "Four, five.",
    "cleanup_context_before": "Testing 1, 2, 3",
    "cleanup_window": "WhatsApp (WhatsApp.Root.exe)",
}


def _diagnostics(tmp_path: Path, *, capture: bool = True) -> SessionDiagnostics:
    paths = AppPaths(base=tmp_path)
    paths.ensure_dirs()
    session_dir = paths.diagnostics_dir / _SESSION
    session_dir.mkdir()
    paths.activate_session(session_dir)
    config = FakeConfig({"diagnostics.capture_content": capture, "diagnostics.enabled": True})
    return SessionDiagnostics(paths, config, _REGISTRY, _HARDWARE)


def _dictate(diagnostics: SessionDiagnostics, heard: str, typed: str | None, at: datetime = _START) -> None:
    diagnostics.transcription_finished(
        AudioSegment(data=np.zeros(1600, dtype=np.float32), sample_rate=16000),
        TranscriptionResult(text=heard, language="en", language_probability=0.9, transcription_seconds=0.4),
        at,
        trace=None if typed is None else DictationTrace(typed=typed),
    )


def _log(tmp_path: Path) -> Path:
    return tmp_path / "diagnostics" / _SESSION / "session.jsonl"


def _logged(tmp_path: Path) -> list[dict]:
    lines = _log(tmp_path).read_text(encoding="utf-8").splitlines()
    return [event for event in map(json.loads, lines) if event.get("event") == "transcription"]


def _as_if_logged_with_the_old_cleanup_model(tmp_path: Path) -> None:
    lines = _log(tmp_path).read_text(encoding="utf-8").splitlines()
    rewritten = []
    for line in lines:
        event = json.loads(line)
        if event.get("event") == "transcription":
            event.update(_OLD_CLEANUP_FIELDS)
        rewritten.append(json.dumps(event, ensure_ascii=False))
    _log(tmp_path).write_text("\n".join(rewritten) + "\n", encoding="utf-8")


class TestWhatTheSessionLogKeeps:
    def test_the_typed_text_is_saved_while_history_is_on(self, tmp_path):
        _dictate(_diagnostics(tmp_path), "Four five", "Testing 1, 2, 3, 4, 5.")

        entry = load_history(tmp_path / "diagnostics")[0]

        assert entry.text == "Four five"
        assert entry.typed == "Testing 1, 2, 3, 4, 5."

    def test_with_history_off_no_text_at_all_is_kept(self, tmp_path):
        _dictate(_diagnostics(tmp_path, capture=False), "Four five", "Testing 1, 2, 3, 4, 5.")

        event = _logged(tmp_path)[0]

        assert TYPED_FIELD not in event
        assert "Four" not in json.dumps(event)

    def test_a_dictation_with_no_trace_has_no_typed_text(self, tmp_path):
        _dictate(_diagnostics(tmp_path), "make it shorter", None)

        assert load_history(tmp_path / "diagnostics")[0].typed == ""

    def test_unicode_and_line_breaks_survive_the_round_trip(self, tmp_path):
        typed = "Steps:\n1. Café in 東京\n2. Say مرحبا to José 🎉"
        _dictate(_diagnostics(tmp_path), "steps cafe in tokyo", typed)

        assert load_history(tmp_path / "diagnostics")[0].typed == typed


class TestHowHistoryShowsIt:
    def test_the_list_shows_what_was_typed_and_search_finds_either_version(self, tmp_path):
        _dictate(_diagnostics(tmp_path), "Four five", "Testing 1, 2, 3, 4, 5.")
        entries = load_history(tmp_path / "diagnostics")

        assert preview(entries[0]) == "Testing 1, 2, 3, 4, 5."
        for needle in ("four five", "4, 5"):
            assert filter_entries(entries, needle) == entries

    def test_an_entry_from_before_typed_text_was_kept_still_shows_its_transcription(self, tmp_path):
        _dictate(_diagnostics(tmp_path), "An old dictation", None)

        assert preview(load_history(tmp_path / "diagnostics")[0]) == "An old dictation"

    def test_an_entry_logged_with_the_old_cleanup_model_loads_and_shows_its_typed_text(self, tmp_path):
        _dictate(_diagnostics(tmp_path), "Four five", "Testing 1, 2, 3, 4, 5.")
        _as_if_logged_with_the_old_cleanup_model(tmp_path)

        entries = load_history(tmp_path / "diagnostics")

        assert [(entry.text, entry.typed) for entry in entries] == [("Four five", "Testing 1, 2, 3, 4, 5.")]
        assert filter_entries(entries, "Four, five.") == []


class TestDeleting:
    def test_the_typed_text_and_the_old_cleanup_text_are_all_blanked(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        _dictate(diagnostics, "Four five", "Testing 1, 2, 3, 4, 5.")
        _dictate(diagnostics, "Keep me", "Keep me too.")
        _as_if_logged_with_the_old_cleanup_model(tmp_path)
        doomed = next(entry for entry in load_history(tmp_path / "diagnostics") if entry.text == "Four five")

        delete_entries(tmp_path / "diagnostics", [doomed], _START + timedelta(days=1))

        deleted, kept = _logged(tmp_path)
        text_fields = ("text", TYPED_FIELD, "cleanup_answer", "cleanup_context_before", "cleanup_window")
        assert all(deleted[field] == "" for field in text_fields)
        assert kept[TYPED_FIELD] == "Keep me too."
        assert kept["cleanup_answer"] == "Four, five."
        assert [entry.text for entry in load_history(tmp_path / "diagnostics")] == ["Keep me"]


class TestTheSpreadsheetExport:
    def test_thousands_of_dictations_export_with_both_versions_and_nothing_runs_as_a_formula(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        for index in range(2000):
            _dictate(
                diagnostics,
                f"equals to {index}" if index % 2 else f"dictation {index} café",
                f"= {index}" if index % 2 else f"Dictation {index} café.",
                _START + timedelta(seconds=index),
            )
        destination = tmp_path / "history.csv"

        rows_written = export_csv(load_history(tmp_path / "diagnostics"), destination)

        assert destination.read_bytes().startswith(b"\xef\xbb\xbf")
        with destination.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        header, body = rows[0], rows[1:]
        assert rows_written == len(body) == 2000
        assert header == ["Time", "Status", "Audio seconds", "Transcription", "Typed"]
        assert body[0][3:] == ["equals to 1999", "'= 1999"]
        assert body[1][3:] == ["dictation 1998 café", "Dictation 1998 café."]

    def test_an_entry_without_typed_text_exports_with_that_cell_blank(self, tmp_path):
        _dictate(_diagnostics(tmp_path), "", None)
        destination = tmp_path / "history.csv"

        export_csv(load_history(tmp_path / "diagnostics"), destination)

        with destination.open(encoding="utf-8-sig", newline="") as stream:
            row = list(csv.reader(stream))[1]
        assert row[3:] == ["", ""]
