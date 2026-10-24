import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from skryb.core import json_config_adapter
from skryb.core.config import ConfigurationStore
from skryb.core.json_config_adapter import JsonConfigAdapter


def _history_entries(config_path: Path) -> list[dict[str, object]]:
    history_path = config_path.with_name("config-history.jsonl")
    return [
        cast(dict[str, object], json.loads(line))
        for line in history_path.read_text(encoding="utf-8").splitlines()
    ]


class TestJsonConfigAdapterDefaults:
    def test_should_return_default_for_unknown_key(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        assert config.get("nonexistent", "fallback") == "fallback"

    def test_should_return_none_when_no_default_given(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        assert config.get("nonexistent") is None

    def test_should_provide_default_config_on_first_load(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        assert config.get("hotkey.bindings") == [{"keys": "ctrl+alt+m", "mode": "toggle", "action": "dictate"}]
        assert config.get("hotkey.double_tap_locks_hold") is True
        assert config.get("output.mode") == "direct_paste"
        assert config.get("transcription.model_size") is None
        assert config.get("overlay.enabled") is True
        assert config.get("transcription.device") == "auto"
        assert config.get("audio.sample_rate") == 16000
        assert config.get("diagnostics.enabled") is True
        assert config.get("diagnostics.retain_sessions") == 50


class TestJsonConfigAdapterDotNotation:
    def test_should_get_nested_value_with_dot_notation(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        assert config.get("transcription.beam_size") == 5

    def test_should_set_nested_value_with_dot_notation(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        config.set("transcription.model_size", "nemotron-speech-streaming-en-0.6b")

        assert config.get("transcription.model_size") == "nemotron-speech-streaming-en-0.6b"

    def test_should_get_top_level_section_as_dict(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        hotkey = config.get("hotkey")

        assert isinstance(hotkey, dict)
        assert hotkey["double_tap_locks_hold"] is True


class TestJsonConfigAdapterPersistence:
    def test_should_distinguish_missing_profile_default_from_persisted_model(
        self, tmp_path,
    ):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)

        assert config.has_persisted_value("transcription.model_size") is False

        config.load()
        config.set("transcription.model_size", "parakeet-tdt-0.6b-v3")
        config.save()

        assert config.has_persisted_value("transcription.model_size") is True

    def test_should_persist_changes_to_file(self, tmp_path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        config.set("hotkey.mode", "push_to_talk")

        config.save()

        raw = json.loads(path.read_text())
        assert raw["hotkey"]["mode"] == "push_to_talk"

    def test_should_reload_persisted_changes(self, tmp_path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        config.set("output.mode", "clipboard_only")
        config.save()

        config2 = JsonConfigAdapter(path)
        config2.load()

        assert config2.get("output.mode") == "clipboard_only"

    def test_should_create_file_with_defaults_on_first_load(self, tmp_path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        config.save()

        assert path.exists()
        raw = json.loads(path.read_text())
        assert "hotkey" in raw
        assert "transcription" in raw


class TestJsonConfigAdapterHistory:
    def test_should_record_generic_nested_changes_with_session_and_version(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def current_session_id() -> str:
            return "session-123"

        def current_version(_distribution: str) -> str:
            return "9.8.7"

        monkeypatch.setattr(json_config_adapter, "session_id", current_session_id)
        monkeypatch.setattr(json_config_adapter, "version", current_version)
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        config.save()
        config.set("future.nested.option", "enabled")

        config.save()

        entry = _history_entries(path)[-1]
        assert set(entry) == {"at", "session", "app_version", "changes"}
        assert entry["session"] == "session-123"
        assert entry["app_version"] == "9.8.7"
        recorded_at = datetime.fromisoformat(str(entry["at"]))
        assert recorded_at.tzinfo is UTC
        assert entry["changes"] == [
            {
                "key": "future.nested.option",
                "from": None,
                "to": "enabled",
            }
        ]

    def test_should_skip_history_when_config_is_unchanged(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        config.save()

        config.save()

        assert len(_history_entries(path)) == 1

    def test_should_retain_only_the_newest_history_entries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(json_config_adapter, "_HISTORY_LINE_LIMIT", 3)
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        for sample_rate in range(16_000, 16_005):
            config.set("audio.sample_rate", sample_rate)
            config.save()

        entries = _history_entries(path)
        assert len(entries) == 3
        assert entries[-1]["changes"] == [
            {"key": "audio.sample_rate", "from": 16_003, "to": 16_004}
        ]

    def test_should_save_config_when_history_write_fails(
        self,
        tmp_path: Path,
        app_caplog: pytest.LogCaptureFixture,
    ) -> None:
        path = tmp_path / "config.json"
        history_path = path.with_name("config-history.jsonl")
        history_path.mkdir()
        config = JsonConfigAdapter(path)
        config.load()
        config.set("hotkey.mode", "push_to_talk")

        config.save()

        assert json.loads(path.read_text(encoding="utf-8"))["hotkey"]["mode"] == (
            "push_to_talk"
        )
        assert "Config history write failed" in app_caplog.text


class TestJsonConfigAdapterGetAll:
    def test_should_return_entire_config(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        all_config = config.get_all()

        assert "hotkey" in all_config
        assert "transcription" in all_config
        assert "output" in all_config
        assert "audio" in all_config


class TestJsonConfigAdapterIsAConfigurationStore:
    def test_should_implement_configuration_store_interface(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")

        assert isinstance(config, ConfigurationStore)
