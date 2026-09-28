"""History keeps every version of a dictation: heard, answered by the writing model, typed."""

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.fakes.config import FakeConfig
from tests.fakes.hardware import FakeHardwareDetector
from voicetotext.core.paths import AppPaths
from voicetotext.diagnostics.session import SessionDiagnostics
from voicetotext.history.entries import (
    TRACK_RECORD_SPAN,
    filter_entries,
    load_history,
    preview,
    writing_track_record,
)
from voicetotext.history.management import delete_entries, export_csv
from voicetotext.output.dictation_trace import (
    APPLIED,
    REJECTED,
    SKIPPED,
    TOO_SLOW,
    TRACE_TEXT_FIELDS,
    DictationTrace,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine_factory import build_engine_registry
from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.result import TranscriptionResult

_SESSION = "20260928T134506Z-91220cb9"
_START = datetime(2026, 9, 28, 13, 45, 6, tzinfo=UTC)
_HARDWARE = FakeHardwareDetector(HardwareProfile(kind="cpu", device_name="Test CPU", gfx_arch=""))
_REGISTRY = build_engine_registry()


def _diagnostics(tmp_path: Path, *, capture: bool = True) -> SessionDiagnostics:
    paths = AppPaths(base=tmp_path)
    paths.ensure_dirs()
    session_dir = paths.diagnostics_dir / _SESSION
    session_dir.mkdir()
    paths.activate_session(session_dir)
    config = FakeConfig({"diagnostics.capture_content": capture, "diagnostics.enabled": True})
    return SessionDiagnostics(paths, config, _REGISTRY, _HARDWARE)


def _dictate(diagnostics: SessionDiagnostics, heard: str, trace: DictationTrace | None, at: datetime = _START) -> None:
    diagnostics.transcription_finished(
        AudioSegment(data=np.zeros(1600, dtype=np.float32), sample_rate=16000),
        TranscriptionResult(text=heard, language="en", language_probability=0.9, transcription_seconds=0.4),
        at,
        trace=trace,
    )


def _trace(**fields: object) -> DictationTrace:
    trace = DictationTrace(
        typed="Testing 1, 2, 3, 4, 5.",
        cleanup=APPLIED,
        cleanup_answer="Four, five.",
        cleanup_seconds=0.42,
        context_before="Testing 1, 2, 3",
        window="WhatsApp (WhatsApp.Root.exe)",
    )
    for name, value in fields.items():
        setattr(trace, name, value)
    return trace


def _logged(tmp_path: Path) -> list[dict]:
    lines = (tmp_path / "diagnostics" / _SESSION / "session.jsonl").read_text(encoding="utf-8").splitlines()
    return [event for event in map(json.loads, lines) if event.get("event") == "transcription"]


class TestWhatTheSessionLogKeeps:
    def test_every_version_is_saved_while_history_is_on(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        _dictate(diagnostics, "Four five", _trace())

        entry = load_history(tmp_path / "diagnostics")[0]

        assert entry.text == "Four five"
        assert entry.writing is not None
        assert entry.writing.outcome == APPLIED
        assert entry.writing.answer == "Four, five."
        assert entry.writing.typed == "Testing 1, 2, 3, 4, 5."
        assert entry.writing.seconds == pytest.approx(0.42)
        assert entry.writing.context_before == "Testing 1, 2, 3"
        assert entry.writing.window == "WhatsApp (WhatsApp.Root.exe)"

    def test_with_history_off_only_the_outcome_is_kept_and_no_text_at_all(self, tmp_path):
        diagnostics = _diagnostics(tmp_path, capture=False)
        _dictate(diagnostics, "Four five", _trace(cleanup=REJECTED, cleanup_reason="a number changed"))

        event = _logged(tmp_path)[0]

        assert event["cleanup"] == REJECTED
        assert event["cleanup_reason"] == "a number changed"
        assert not any(field in event for field in TRACE_TEXT_FIELDS)
        assert "Four" not in json.dumps(event)

    def test_a_dictation_with_no_trace_logs_no_writing_fields(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        _dictate(diagnostics, "make it shorter", None)

        assert load_history(tmp_path / "diagnostics")[0].writing is None

    def test_unicode_and_line_breaks_survive_the_round_trip(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        typed = "Steps:\n1. Café in 東京\n2. Say مرحبا to José 🎉"
        _dictate(diagnostics, "steps cafe in tokyo", _trace(typed=typed, cleanup_answer=typed))

        assert load_history(tmp_path / "diagnostics")[0].writing.typed == typed


class TestHowHistoryShowsIt:
    def test_the_list_shows_what_was_typed_and_search_finds_any_version(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        _dictate(diagnostics, "Four five", _trace())
        entries = load_history(tmp_path / "diagnostics")

        assert preview(entries[0]) == "Testing 1, 2, 3, 4, 5."
        for needle in ("four five", "4, 5", "FOUR, FIVE."):
            assert filter_entries(entries, needle) == entries

    def test_an_entry_from_before_versions_were_kept_still_shows_its_transcription(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        _dictate(diagnostics, "An old dictation", None)

        assert preview(load_history(tmp_path / "diagnostics")[0]) == "An old dictation"


class TestDeletingRemovesEveryVersion:
    def test_the_answer_the_typed_text_and_the_cursor_text_are_all_blanked(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        _dictate(diagnostics, "Four five", _trace())
        _dictate(diagnostics, "Keep me", _trace(typed="Keep me too."))
        doomed = next(entry for entry in load_history(tmp_path / "diagnostics") if entry.text == "Four five")

        delete_entries(tmp_path / "diagnostics", [doomed], _START + timedelta(days=1))

        deleted, kept = _logged(tmp_path)
        assert all(deleted[field] == "" for field in ("text", *TRACE_TEXT_FIELDS))
        assert deleted["cleanup"] == APPLIED
        assert kept["typed_text"] == "Keep me too."
        assert [entry.text for entry in load_history(tmp_path / "diagnostics")] == ["Keep me"]


class TestTheSpreadsheetExport:
    def test_thousands_of_dictations_export_with_every_version_and_nothing_runs_as_a_formula(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        for index in range(2000):
            _dictate(
                diagnostics,
                f"equals to {index}" if index % 2 else f"dictation {index} café",
                _trace(typed=f"= {index}" if index % 2 else f"Dictation {index} café.", cleanup_answer="@admin +1 -1"),
                _START + timedelta(seconds=index),
            )
        destination = tmp_path / "history.csv"

        rows_written = export_csv(load_history(tmp_path / "diagnostics"), destination)

        assert destination.read_bytes().startswith(b"\xef\xbb\xbf")
        with destination.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
        header, body = rows[0], rows[1:]
        assert rows_written == len(body) == 2000
        assert header[4:10] == ["Transcription", "Writing model", "Why not used", "Writing model answer", "Model seconds", "Typed"]
        newest = body[0]
        assert newest[4] == "equals to 1999"
        assert newest[9] == "'= 1999"
        assert newest[7] == "'@admin +1 -1"
        assert body[1][9] == "Dictation 1998 café."

    def test_an_entry_without_versions_exports_with_those_cells_blank(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        _dictate(diagnostics, "", None)
        destination = tmp_path / "history.csv"

        export_csv(load_history(tmp_path / "diagnostics"), destination)

        with destination.open(encoding="utf-8-sig", newline="") as stream:
            row = list(csv.reader(stream))[1]
        assert row[4:] == [""] * 8


class TestTheTrackRecord:
    def test_it_counts_the_newest_dictations_the_model_was_asked_about_and_why_it_failed(self, tmp_path):
        diagnostics = _diagnostics(tmp_path)
        outcomes = (
            [(APPLIED, "")] * 30
            + [(REJECTED, "a number changed")] * 8
            + [(REJECTED, "a hedge was dropped")] * 3
            + [(TOO_SLOW, "it took 4.20s, over the 4.0s limit")] * 5
            + [(SKIPPED, "the writing model is busy")] * 4
            + [("", "")] * 10
        )
        for index, (outcome, reason) in enumerate(outcomes):
            _dictate(diagnostics, f"dictation {index}", _trace(cleanup=outcome, cleanup_reason=reason), _START + timedelta(seconds=index))
        for index in range(20):  # older than the span, never counted
            _dictate(diagnostics, f"old {index}", _trace(cleanup=REJECTED, cleanup_reason="old"), _START - timedelta(days=1, seconds=index))

        record = writing_track_record(load_history(tmp_path / "diagnostics"))

        assert record.dictations == TRACK_RECORD_SPAN
        assert record.outcomes == {APPLIED: 30, REJECTED: 11, TOO_SLOW: 5, SKIPPED: 4}
        assert record.rejections == [("a number changed", 8), ("a hedge was dropped", 3)]

    def test_no_dictations_gives_an_empty_record(self):
        record = writing_track_record([])

        assert (record.dictations, record.outcomes, record.rejections) == (0, {}, [])
