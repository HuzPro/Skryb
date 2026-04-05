import copy
import json
import threading
from pathlib import Path
from typing import Any

from voicetotext.core.config import ConfigurationStore

DEFAULT_CONFIG = {
    "hotkey": {
        "mode": "toggle",
        "combination": "ctrl+alt+m",
    },
    "output": {
        "mode": "auto_paste",
    },
    "transcription": {
        "model_size": "base",
        "device": "auto",
        "compute_type": "float16",
        "language": None,
        "beam_size": 5,
        "vad_filter": True,
    },
    "audio": {
        "sample_rate": 16000,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class JsonConfigAdapter(ConfigurationStore):

    def __init__(self, path: Path):
        self._path = Path(path)
        self._data: dict = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                self._data = _deep_merge(DEFAULT_CONFIG, raw)
            else:
                self._data = copy.deepcopy(DEFAULT_CONFIG)

    def save(self) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            parts = key.split(".")
            current = self._data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            parts = key.split(".")
            current = self._data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

    def get_all(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._data)
