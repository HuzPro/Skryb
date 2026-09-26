import json
import os
import threading
from collections.abc import Iterable
from pathlib import Path

# One lock per file, shared by every writer in the process: history
# deletion appends to the session log the diagnostics writer is still
# appending to, and two handles in append mode can overwrite each other.
_FILE_LOCKS: dict[str, threading.Lock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(path))
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(key, threading.Lock())


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
