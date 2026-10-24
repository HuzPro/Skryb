import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

from skryb.output.dictation_trace import (
    ANSWER_FIELD,
    CONTEXT_FIELD,
    OUTCOME_FIELD,
    REASON_FIELD,
    REJECTED,
    SECONDS_FIELD,
    TYPED_FIELD,
    WINDOW_FIELD,
)

# Matches the folder names ``startup_log`` gives sessions. The ownership
# marker is not required: sessions from before it existed still hold
# dictations worth listing, and history only ever reads them.
_SESSION_NAME = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")
_SESSION_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
_PREVIEW_CHARS = 120

CANCELLED = "Cancelled"
FAILED = "Failed"
EMPTY = "Empty"
NOT_SAVED = "Not saved"

_CAPTURE_HINT = (
    "Dictations are listed here when \"Save history\" is on in "
    "Settings > History."
)
_DELETED_EVENT = "history_deleted"


@dataclass(frozen=True)
class WritingRecord:
    """What the writing model made of a dictation, and what was typed in the end.

    ``outcome`` is empty when the model was not asked (cleanup off). The
    text fields are empty in a log kept with content capture off, and in
    dictations from before they were recorded.
    """

    outcome: str
    reason: str
    seconds: float | None
    answer: str
    typed: str
    context_before: str
    window: str


@dataclass(frozen=True)
class HistoryEntry:
    """One dictation or cancelled recording read back from a session log.

    ``key`` is stable across re-reads of the same append-only log, so a
    selection survives a refresh. ``audio_path`` is where the recording
    should be; the file itself may have been removed since. ``seq`` is the
    number the session gave the recording, which a deletion names.
    """

    key: tuple[str, int]
    at: datetime
    text: str
    audio_seconds: float
    audio_path: Path | None
    status: str
    error: str | None
    seq: int | None = None
    transcript_path: Path | None = None
    # "edit" for a voice edit, whose text is the spoken instruction.
    kind: str = "dictation"
    writing: WritingRecord | None = None

    @property
    def is_cancelled(self) -> bool:
        return self.status == CANCELLED

    @property
    def shown_text(self) -> str:
        """What was typed when that was saved, else what the engine heard."""
        return self.writing.typed if self.writing is not None and self.writing.typed else self.text


def load_history(diagnostics_dir: Path) -> list[HistoryEntry]:
    """Every dictation and cancelled recording under *diagnostics_dir*, newest first.

    Unreadable sessions and malformed lines (a line still being written
    included) are skipped rather than failing the whole history.
    """
    try:
        session_dirs = [
            path for path in diagnostics_dir.iterdir()
            if _SESSION_NAME.fullmatch(path.name)
        ]
    except OSError:
        return []
    entries: list[HistoryEntry] = []
    for session_dir in session_dirs:
        entries.extend(_session_entries(session_dir))
    entries.sort(key=lambda entry: (entry.at, entry.key), reverse=True)
    return entries


def find_entry(diagnostics_dir: Path, key: tuple[str, int]) -> HistoryEntry | None:
    """The entry *key* names, reading only its own session's log.

    A key that does not name a session folder finds nothing, so it can
    never lead a caller outside *diagnostics_dir*.
    """
    session, _line = key
    if not _SESSION_NAME.fullmatch(session):
        return None
    return next(
        (entry for entry in _session_entries(diagnostics_dir / session) if entry.key == key),
        None,
    )


def filter_entries(entries: list[HistoryEntry], query: str) -> list[HistoryEntry]:
    needle = query.strip().casefold()
    if not needle:
        return entries
    return [entry for entry in entries if any(needle in text.casefold() for text in _searched(entry))]


TRACK_RECORD_SPAN = 50


@dataclass(frozen=True)
class TrackRecord:
    """How the writing model fared on the latest dictations it was asked about."""

    dictations: int
    outcomes: dict[str, int]
    # Why answers were rejected, most common first.
    rejections: list[tuple[str, int]]


def writing_track_record(
    entries: list[HistoryEntry], span: int = TRACK_RECORD_SPAN,
) -> TrackRecord:
    """Outcomes of the newest *span* dictations the model was asked about; *entries* newest first."""
    asked = [
        entry.writing for entry in entries
        if entry.writing is not None and entry.writing.outcome
    ][:span]
    outcomes = Counter(record.outcome for record in asked)
    rejections = Counter(record.reason for record in asked if record.outcome == REJECTED)
    return TrackRecord(len(asked), dict(outcomes), rejections.most_common())


def _searched(entry: HistoryEntry) -> tuple[str, ...]:
    if entry.writing is None:
        return (entry.text,)
    return (entry.text, entry.writing.typed, entry.writing.answer)


def preview(entry: HistoryEntry) -> str:
    text = " ".join(entry.shown_text.split())
    if not text:
        if entry.error:
            return f"({entry.error})"
        return "(cancelled - not transcribed)" if entry.status == CANCELLED else ""
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[: _PREVIEW_CHARS - 1] + "…"


def local_time(entry: HistoryEntry) -> str:
    return entry.at.astimezone().strftime("%Y-%m-%d %H:%M")


def empty_state_message(
    entries: list[HistoryEntry], diagnostics_dir: Path, *, saving: bool,
) -> str:
    """What to tell the user when history has nothing useful to show, else ``""``.

    The hint to turn history saving on is left out while *saving* is on.
    """
    hint = "" if saving else f" {_CAPTURE_HINT}"
    if not entries:
        where = (
            "No dictations yet." if diagnostics_dir.is_dir()
            else f"The logs folder {diagnostics_dir} does not exist yet."
        )
        return f"{where}{hint}"
    if entries[0].status == NOT_SAVED:
        return f"Recent dictations were not saved.{hint}"
    return ""


def _session_entries(session_dir: Path) -> list[HistoryEntry]:
    try:
        started_at = datetime.strptime(
            session_dir.name[:16], _SESSION_TIME_FORMAT,
        ).replace(tzinfo=UTC)
        with (session_dir / "session.jsonl").open(
            encoding="utf-8", errors="replace",
        ) as stream:
            lines = stream.readlines()
    except (OSError, ValueError):
        return []
    events = [
        (line_number, event)
        for line_number, event in enumerate(map(_parsed, lines))
        if event is not None
    ]
    deleted = {
        _seq(event.get("seq")) for _line_number, event in events
        if event.get("event") == _DELETED_EVENT
    }
    deleted.discard(None)
    entries = []
    for line_number, event in events:
        entry = _entry(event, session_dir, line_number, started_at)
        if entry is not None and entry.seq not in deleted:
            entries.append(entry)
    return entries


def _parsed(line: str) -> dict | None:
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _entry(
    event: dict,
    session_dir: Path,
    line_number: int,
    started_at: datetime,
) -> HistoryEntry | None:
    kind = event.get("event")
    if event.get("history_deleted") is True:
        return None
    transcript = _session_file(session_dir, event.get("transcript_path"))
    if kind == "transcription":
        at = _timestamp(event.get("requested_at"), started_at)
        text = _text(event, transcript)
        if _text_is_gone(event, text, transcript):
            return None
    elif kind == "cancelled":
        at = _timestamp(event.get("at"), started_at)
        text = ""
    else:
        return None
    error = event.get("error")
    error = error.strip() if isinstance(error, str) and error.strip() else None
    return HistoryEntry(
        key=(session_dir.name, line_number),
        at=at,
        text=text,
        audio_seconds=_seconds(event.get("audio_seconds")),
        audio_path=_session_file(session_dir, event.get("audio_path")),
        status=_status(kind, text, error, event.get("is_empty")),
        error=error,
        seq=_seq(event.get("index")),
        transcript_path=transcript,
        kind="edit" if event.get("kind") == "edit" else "dictation",
        writing=_writing(event) if kind == "transcription" else None,
    )


def _writing(event: dict) -> WritingRecord | None:
    """The trace a dictation logged; a line from before traces has none."""
    if OUTCOME_FIELD not in event and TYPED_FIELD not in event:
        return None
    seconds = event.get(SECONDS_FIELD)
    return WritingRecord(
        outcome=_string(event.get(OUTCOME_FIELD)),
        reason=_string(event.get(REASON_FIELD)),
        seconds=_seconds(seconds) if seconds is not None else None,
        answer=_string(event.get(ANSWER_FIELD)),
        typed=_string(event.get(TYPED_FIELD)),
        context_before=_string(event.get(CONTEXT_FIELD)),
        window=_string(event.get(WINDOW_FIELD)),
    )


def _string(value: object) -> str:
    return value if isinstance(value, str) else ""


def _text_is_gone(event: dict, text: str, transcript: Path | None) -> bool:
    """A result with words whose saved transcript has since been deleted."""
    return (
        not text.strip()
        and event.get("is_empty") is False
        and transcript is not None
        and not transcript.is_file()
    )


def _seq(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _status(kind: str, text: str, error: str | None, is_empty: object) -> str:
    if kind == "cancelled":
        return CANCELLED
    if error is not None:
        return FAILED
    if text.strip():
        return ""
    # With content capture off the log keeps ``is_empty`` for the real
    # result but blanks the text, so a non-empty result lost its text.
    return NOT_SAVED if is_empty is False else EMPTY


def _text(event: dict, transcript: Path | None) -> str:
    text = event.get("text")
    if isinstance(text, str) and text.strip():
        return text
    # The log line and the transcript file are written from the same
    # string, so the file only matters when the line lost its copy.
    if transcript is None:
        return ""
    try:
        return transcript.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _session_file(session_dir: Path, relative: object) -> Path | None:
    if not isinstance(relative, str) or not relative.strip():
        return None
    path = Path(relative.strip())
    # Judged under both path flavours: "C:/Windows/x.wav" is relative on Linux.
    anchored = PureWindowsPath(path).anchor or PurePosixPath(path).anchor
    if anchored or ".." in path.parts:
        return None
    return session_dir / path


def _timestamp(value: object, fallback: datetime) -> datetime:
    if not isinstance(value, str):
        return fallback
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return fallback
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value) if math.isfinite(value) and value > 0 else 0.0
