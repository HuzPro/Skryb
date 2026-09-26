import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

# One lock per file, shared by every writer in the process: history
# deletion rewrites the session log the diagnostics writer is still
# appending to, and two handles in append mode can overwrite each other.
_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()

# A reader holding the file open (the history window loading it, a virus
# scanner) makes Windows refuse the replace for a moment.
_REPLACE_ATTEMPTS = 20
_REPLACE_RETRY_SECONDS = 0.05


def _lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(path))
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(key, threading.Lock())


def encode_line(event: dict[str, object], ending: bytes = os.linesep.encode()) -> bytes:
    """One event as :meth:`JsonlWriter.append` writes it, line ending included.

    Appends go through a text-mode file, so on Windows each line ends CRLF.
    """
    return json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode() + ending


class JsonlWriter:
    """Appends one JSON object per line, safely from several threads."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = _lock_for(path)

    def append(self, event: dict[str, object]) -> None:
        self.append_many([event])

    def append_many(self, events: Iterable[dict[str, object]]) -> None:
        """Append every event in one write, so they land together or not at all."""
        lines = "".join(
            json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
            for event in events
        )
        if not lines:
            return
        with self._lock, self._path.open("a", encoding="utf-8") as stream:
            stream.write(lines)

    def rewrite(self, edit: Callable[[list[bytes]], list[bytes]]) -> None:
        """Replace the file with *edit* applied to its lines, atomically.

        Held under the same lock as :meth:`append`, so no line can land
        between the read and the replace. The new content is written to a
        temporary file beside the old one and swapped in with one rename: a
        crash leaves the old file whole plus a stray ``.tmp`` nothing reads.
        """
        with self._lock:
            lines = self._path.read_bytes().splitlines(keepends=True)
            temporary = self._write_temporary(b"".join(edit(lines)))
            try:
                _replace_patiently(temporary, self._path)
            except OSError:
                temporary.unlink(missing_ok=True)
                raise

    def _write_temporary(self, content: bytes) -> Path:
        handle, name = tempfile.mkstemp(
            dir=self._path.parent, prefix=f"{self._path.name}.", suffix=".tmp",
        )
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        return Path(name)


def _replace_patiently(source: Path, target: Path) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_RETRY_SECONDS)
