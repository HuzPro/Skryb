import json
import threading
from pathlib import Path


class JsonlWriter:
    """Appends one JSON object per line, safely from several threads."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def append(self, event: dict[str, object]) -> None:
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
