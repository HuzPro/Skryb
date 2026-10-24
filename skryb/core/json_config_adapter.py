import copy
import json
import logging
import threading
from collections import deque
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from skryb.core.config import ConfigurationStore, ConfigValue
from skryb.core.startup_log import session_id

_HISTORY_FILE_NAME = "config-history.jsonl"
_HISTORY_LINE_LIMIT = 200
_logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "hotkey": {
        "bindings": [
            {"keys": "ctrl+alt+m", "mode": "toggle", "action": "dictate"},
        ],
        "double_tap_locks_hold": True,
    },
    "output": {
        "mode": "direct_paste",
        "paste_method_overrides": {},
    },
    "overlay": {
        "enabled": True,
        "position": {"x": 0.5, "y": 1.0},
    },
    "transcription": {
        "device": "auto",
        "language": "en",
        "beam_size": 5,
        "mode": "batch",
        "crisperwhisper_mode": "intended",
        "wake_after_idle_seconds": 60.0,
    },
    "audio": {
        "sample_rate": 16000,
        "warn_when_silent": True,
        "preferred_devices": [],
        "cues_enabled": True,
        "cue_volume": 0.6,
        "duck_other_audio": "off",
        "duck_level": 0.2,
    },
    "recording": {
        "max_duration_seconds": 600.0,
        "warn_before_cap_seconds": 30.0,
    },
    "diagnostics": {
        "enabled": True,
        "capture_content": False,
        "retain_sessions": 50,
        "directory": "",
    },
    "writing": {
        "cleanup": "light",
        "lists": True,
        "deadline_seconds": 4.0,
        "context": True,
        "learn_words": True,
        "learn_style": True,
        "style_notes": "",
        "after_dictation_transform": "",
        "runtime_path": "",
        "model_path": "",
    },
    "history": {
        "retention_days": 0,
    },
}


def load_json_object(path: Path) -> dict[str, object]:
    raw = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise TypeError(f"'{path}' must contain a JSON object")
    return cast(dict[str, object], raw)


def deep_merge_config(
    base: dict[str, object],
    override: dict[str, object],
) -> dict[str, object]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_config(
                cast(dict[str, object], result[key]),
                cast(dict[str, object], value),
            )
        else:
            result[key] = copy.deepcopy(value)
    return result


def _flatten_config(
    data: dict[str, object],
    prefix: str = "",
) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in data.items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            flattened.update(
                _flatten_config(cast(dict[str, object], value), dotted_key)
            )
        else:
            flattened[dotted_key] = value
    return flattened


def _config_changes(
    before: dict[str, object],
    after: dict[str, object],
) -> list[dict[str, object]]:
    previous = _flatten_config(before)
    updated = _flatten_config(after)
    changes: list[dict[str, object]] = []
    for key in sorted(previous.keys() | updated.keys()):
        if (
            key in previous
            and key in updated
            and json.dumps(previous[key], sort_keys=True, ensure_ascii=False)
            == json.dumps(updated[key], sort_keys=True, ensure_ascii=False)
        ):
            continue
        changes.append({
            "key": key,
            "from": previous.get(key),
            "to": updated.get(key),
        })
    return changes


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _app_version() -> str:
    try:
        return version("skryb")
    except PackageNotFoundError:
        return "unknown"


class JsonConfigAdapter(ConfigurationStore):

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._data: dict[str, object] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._path.exists():
                self._data = deep_merge_config(
                    DEFAULT_CONFIG,
                    load_json_object(self._path),
                )
            else:
                self._data = copy.deepcopy(DEFAULT_CONFIG)

    def has_persisted_value(self, key: str) -> bool:
        if not self._path.exists():
            return False
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        current: object = raw
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return True

    def save(self) -> None:
        with self._lock:
            previous = self._history_baseline()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._record_history(previous)

    def get(self, key: str, default: ConfigValue = None) -> ConfigValue:
        with self._lock:
            parts = key.split(".")
            current = self._data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current

    def set(self, key: str, value: ConfigValue) -> None:
        with self._lock:
            parts = key.split(".")
            current = self._data
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

    def get_all(self) -> dict[str, object]:
        with self._lock:
            return copy.deepcopy(self._data)

    def replace_all(self, data: dict[str, object]) -> None:
        with self._lock:
            self._data = copy.deepcopy(data)

    def _history_baseline(self) -> dict[str, object]:
        if not self._path.exists():
            return {}
        try:
            return load_json_object(self._path)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Config history read failed: %s", exc)
            return copy.deepcopy(self._data)

    def _record_history(self, previous: dict[str, object]) -> None:
        try:
            self._append_history(previous)
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Config history write failed: %s", exc)

    def _append_history(self, previous: dict[str, object]) -> None:
        changes = _config_changes(previous, self._data)
        if not changes:
            return
        entry: dict[str, object] = {
            "at": _utc_now(),
            "session": session_id(),
            "app_version": _app_version(),
            "changes": changes,
        }
        history_path = self._path.with_name(_HISTORY_FILE_NAME)
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        self._trim_history(history_path)

    @staticmethod
    def _trim_history(history_path: Path) -> None:
        with history_path.open("r", encoding="utf-8") as stream:
            newest: deque[str] = deque(
                stream,
                maxlen=_HISTORY_LINE_LIMIT + 1,
            )
        if len(newest) <= _HISTORY_LINE_LIMIT:
            return
        with history_path.open("w", encoding="utf-8") as stream:
            stream.writelines(list(newest)[-_HISTORY_LINE_LIMIT:])
