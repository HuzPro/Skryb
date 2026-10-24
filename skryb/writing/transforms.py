import json
import logging
import threading
from pathlib import Path
from typing import cast

_logger = logging.getLogger(__name__)

BUILTIN_TRANSFORMS: dict[str, str] = {
    "Concise": "Make it concise. Keep every fact, name and number.",
    "Bullet points": "Turn it into short bullet points, one idea per line, each starting with \"- \".",
    "Professional": "Make the tone clear and professional, without adding greetings or new information.",
    "Casual": "Make the tone relaxed and friendly, like a message to a colleague.",
}
MAX_TRANSFORMS = 50
MAX_NAME_CHARS = 60
MAX_INSTRUCTION_CHARS = 1000


class TransformStore:
    """Named rewrite instructions, kept in ``transforms.json``.

    The built-ins are only the starting list: once saved, the file holds the
    whole list, edits and removals included.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._cached: dict[str, str] | None = None

    def all(self) -> dict[str, str]:
        with self._lock:
            if self._cached is None:
                self._cached = self._read()
            return dict(self._cached)

    def instruction(self, name: str) -> str:
        return self.all().get(name, "") if name else ""

    def save(self, transforms: dict[str, str]) -> None:
        cleaned = valid_transforms(transforms)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        staged = self._path.with_suffix(".tmp")
        staged.write_text(
            json.dumps({"transforms": cleaned}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staged.replace(self._path)
        with self._lock:
            self._cached = cleaned

    def _read(self) -> dict[str, str]:
        if not self._path.exists():
            return dict(BUILTIN_TRANSFORMS)
        try:
            document: object = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            _logger.warning("Transformations ignored: %s", exc)
            return dict(BUILTIN_TRANSFORMS)
        stored = cast(dict[str, object], document).get("transforms") if isinstance(document, dict) else None
        if not isinstance(stored, dict):
            _logger.warning("Transformations ignored: no transforms object")
            return dict(BUILTIN_TRANSFORMS)
        return valid_transforms(cast(dict[object, object], stored))


def valid_transforms(transforms: dict[object, object] | dict[str, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for name, instruction in transforms.items():
        if not isinstance(name, str) or not isinstance(instruction, str):
            continue
        name, instruction = name.strip(), instruction.strip()
        if 0 < len(name) <= MAX_NAME_CHARS and 0 < len(instruction) <= MAX_INSTRUCTION_CHARS:
            cleaned[name] = instruction
        if len(cleaned) == MAX_TRANSFORMS:
            break
    return cleaned
