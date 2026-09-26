"""Deleting and exporting history entries.

A deletion removes the entry's recording and transcript, blanks the text
its log line kept (marking the line ``"history_deleted": true``) and adds a
``history_deleted`` line naming its ``seq``, so
:func:`~voicetotext.history.entries.load_history` stops listing it. The log
is rewritten in one atomic swap; every other line stays byte for byte, and
the session's text log and other diagnostics files are never touched.
"""

import json
import logging
import os
import shutil
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

from voicetotext.core.timestamps import utc_iso, utc_now
from voicetotext.diagnostics.jsonl import JsonlWriter, encode_line
from voicetotext.history.entries import HistoryEntry, load_history

_logger = logging.getLogger(__name__)

# The only folders a session keeps per-entry files in. A log line naming
# anything else (``session.jsonl`` included) is never acted on.
_ENTRY_FOLDERS = frozenset({"audio", "transcripts"})
_NEWLINE = os.linesep.encode()


def delete_entries(
    diagnostics_dir: Path, entries: Iterable[HistoryEntry], now: datetime,
) -> int:
    """Delete *entries* and return how many were marked deleted.

    Files that are already gone are fine: the tombstone alone hides the
    entry. A session whose log cannot be rewritten keeps its entries
    listed, so the user can see the deletion did not happen.
    """
    by_session: dict[str, list[HistoryEntry]] = defaultdict(list)
    for entry in entries:
        by_session[entry.key[0]].append(entry)
    deleted = 0
    for session_name, session_entries in by_session.items():
        deleted += _delete_in_session(
            diagnostics_dir / session_name, session_entries, now,
        )
    return deleted


def delete_all_history(diagnostics_dir: Path, now: datetime) -> int:
    return delete_entries(diagnostics_dir, load_history(diagnostics_dir), now)


def delete_all_in_background(
    diagnostics_dir: Path, on_done: Callable[[], None],
) -> None:
    """Delete all history on a daemon thread, then call *on_done* from it."""
    threading.Thread(
        target=_delete_all_then, args=(diagnostics_dir, on_done),
        name="history-delete-all", daemon=True,
    ).start()


def _delete_all_then(diagnostics_dir: Path, on_done: Callable[[], None]) -> None:
    try:
        deleted = delete_all_history(diagnostics_dir, utc_now())
    except Exception:  # a daemon thread would drop it silently
        _logger.exception("Deleting all history failed.")
    else:
        _logger.info("Deleted %d history entries.", deleted)
    on_done()


def export_audio(diagnostics_dir: Path, entry: HistoryEntry, destination: Path) -> None:
    """Copy the entry's recording to *destination*; history is left as it was.

    Raises ``ValueError`` when the log points the recording anywhere but
    its own session folder, and ``OSError`` when the copy fails.
    """
    source = _owned_file(diagnostics_dir / entry.key[0], entry.audio_path)
    if source is None:
        raise ValueError("The recording is not inside its session folder.")
    shutil.copyfile(source, destination)


def default_export_name(entry: HistoryEntry) -> str:
    return f"dictation-{entry.at.astimezone():%Y%m%d-%H%M%S}.wav"


def _delete_in_session(
    session_dir: Path, entries: list[HistoryEntry], now: datetime,
) -> int:
    seqs = sorted({entry.seq for entry in entries if entry.seq is not None})
    for entry in entries:
        for path in (entry.audio_path, entry.transcript_path):
            _remove_owned_file(session_dir, path)
    if not seqs:
        return 0
    try:
        _redact_and_mark_deleted(session_dir, seqs, now)
    except OSError as exc:
        _logger.warning("Could not mark history deleted in %s: %s", session_dir, exc)
        return 0
    return len(seqs)


def _redact_and_mark_deleted(
    session_dir: Path, seqs: list[int], now: datetime,
) -> None:
    at = utc_iso(now)
    tombstones = [
        encode_line({"event": "history_deleted", "seq": seq, "at": at})
        for seq in seqs
    ]

    def edit(lines: list[bytes]) -> list[bytes]:
        deleted = set(seqs)
        redacted = [_redacted_line(line, deleted) for line in lines]
        if redacted and not redacted[-1].endswith(b"\n"):
            redacted[-1] += _NEWLINE  # a line cut short by a crash stays apart
        return redacted + tombstones

    JsonlWriter(session_dir / "session.jsonl").rewrite(edit)


def _redacted_line(line: bytes, seqs: set[int]) -> bytes:
    """*line* with its dictation text blanked if it records a deleted seq."""
    if b'"transcription"' not in line:
        return line
    try:
        event = json.loads(line)
    except ValueError:
        return line
    if not isinstance(event, dict) or event.get("event") != "transcription":
        return line
    index = event.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index not in seqs:
        return line
    if event.get("text") == "" and event.get("history_deleted") is True:
        return line
    event["text"] = ""
    event["history_deleted"] = True
    body = line.rstrip(b"\r\n")
    return encode_line(event, line[len(body):] or _NEWLINE)


def _remove_owned_file(session_dir: Path, path: Path | None) -> None:
    owned = _owned_file(session_dir, path)
    if owned is None:
        return
    try:
        owned.unlink(missing_ok=True)
    except OSError as exc:
        _logger.warning("Could not delete %s: %s", owned, exc)


def _owned_file(session_dir: Path, path: Path | None) -> Path | None:
    """*path* if it is an entry file inside *session_dir*, else ``None``.

    Resolving follows links and junctions, so a folder link inside the
    session cannot lead a deletion out of it.
    """
    if path is None:
        return None
    try:
        root = session_dir.resolve()
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(root):
        return None
    relative = resolved.relative_to(root)
    if len(relative.parts) < 2 or relative.parts[0].casefold() not in _ENTRY_FOLDERS:
        return None
    return resolved
