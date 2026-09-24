import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

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
    "Dictations are listed here when \"Save recordings and transcripts "
    "for diagnostics\" is on in Settings > Output."
)


@dataclass(frozen=True)
class HistoryEntry:
    """One dictation or cancelled recording read back from a session log.

    ``key`` is stable across re-reads of the same append-only log, so a
    selection survives a refresh. ``audio_path`` is where the recording
    should be; the file itself may have been removed since.
    """

    key: tuple[str, int]
    at: datetime
    text: str
    audio_seconds: float
    audio_path: Path | None
    status: str
    error: str | None

    @property
    def is_cancelled(self) -> bool:
        return self.status == CANCELLED


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


def filter_entries(entries: list[HistoryEntry], query: str) -> list[HistoryEntry]:
    needle = query.strip().casefold()
    if not needle:
        return entries
    return [entry for entry in entries if needle in entry.text.casefold()]


def preview(entry: HistoryEntry) -> str:
    text = " ".join(entry.text.split())
    if not text:
        if entry.error:
            return f"({entry.error})"
        return "(cancelled - not transcribed)" if entry.status == CANCELLED else ""
    if len(text) <= _PREVIEW_CHARS:
        return text
    return text[: _PREVIEW_CHARS - 1] + "…"


def local_time(entry: HistoryEntry) -> str:
    return entry.at.astimezone().strftime("%Y-%m-%d %H:%M")


def empty_state_message(entries: list[HistoryEntry], diagnostics_dir: Path) -> str:
    """What to tell the user when history has nothing useful to show, else ``""``."""
    if not entries:
        where = (
            "No dictations yet." if diagnostics_dir.is_dir()
            else f"The logs folder {diagnostics_dir} does not exist yet."
        )
        return f"{where} {_CAPTURE_HINT}"
    if entries[0].status == NOT_SAVED:
        return f"Recent dictations were not saved. {_CAPTURE_HINT}"
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
    entries = []
    for line_number, line in enumerate(lines):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        entry = _entry(event, session_dir, line_number, started_at)
        if entry is not None:
            entries.append(entry)
    return entries


def _entry(
    event: dict,
    session_dir: Path,
    line_number: int,
    started_at: datetime,
) -> HistoryEntry | None:
    kind = event.get("event")
    if kind == "transcription":
        at = _timestamp(event.get("requested_at"), started_at)
        text = _text(event, session_dir)
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
    )


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


def _text(event: dict, session_dir: Path) -> str:
    text = event.get("text")
    if isinstance(text, str) and text.strip():
        return text
    # The log line and the transcript file are written from the same
    # string, so the file only matters when the line lost its copy.
    transcript = _session_file(session_dir, event.get("transcript_path"))
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
    if path.anchor or ".." in path.parts:
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
