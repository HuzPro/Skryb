from pathlib import Path

import pytest

from voicetotext.core.exceptions import ConfigurationError
from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.transcription.model_info import (
    ALL_MODELS,
    DEFAULT_MODEL,
    NEMOTRON_STREAMING_MODELS,
    PARAKEET_MODELS,
    ModelInfo,
    get_model_info,
    is_catalog_model,
    migrate_engine_field,
    migrate_removed_models,
)


_REMOVED_MODEL_EXPECTATIONS = (
    ("tiny", DEFAULT_MODEL),
    ("base", DEFAULT_MODEL),
    ("small", DEFAULT_MODEL),
    ("medium", DEFAULT_MODEL),
    ("large-v3", DEFAULT_MODEL),
    ("large-v3-turbo", DEFAULT_MODEL),
    ("granite-speech-4.1-2b", DEFAULT_MODEL),
    ("parakeet-tdt-0.6b-v2", DEFAULT_MODEL),
    ("nemotron-streaming-compare", "nemotron-speech-streaming-en-0.6b"),
)

_REMOVED_ENGINES = (
    "faster_whisper",
    "transformers",
    "granite_speech",
)


class TestModelInfo:
    def test_should_store_and_format_model_attributes(self) -> None:
        info = ModelInfo(
            name="example", size_mb=1500, description="Example model",
            repo_id="example/model", compatible_engines=("nano_parakeet",),
        )

        assert info.name == "example"
        assert info.repo_id == "example/model"
        assert info.compatible_engines == ("nano_parakeet",)
        assert info.display_size == "1.5 GB"

    def test_compatible_engines_should_default_to_empty(self) -> None:
        info = ModelInfo(name="example", size_mb=1, description="Example")

        assert info.compatible_engines == ()
        assert dict(info.engine_priorities) == {}


class TestModelCatalog:
    def test_should_contain_only_surviving_model_lineups(self) -> None:
        assert set(ALL_MODELS) == (
            set(PARAKEET_MODELS) | set(NEMOTRON_STREAMING_MODELS)
        )

    def test_default_should_be_current_parakeet(self) -> None:
        assert DEFAULT_MODEL == "parakeet-tdt-0.6b-v3"

    def test_should_return_known_model(self) -> None:
        assert get_model_info(DEFAULT_MODEL) is PARAKEET_MODELS[DEFAULT_MODEL]

    def test_should_raise_for_unknown_model(self) -> None:
        with pytest.raises(ConfigurationError, match="unknown-model"):
            get_model_info("unknown-model")

    def test_every_catalog_entry_should_be_visible(self) -> None:
        assert all(is_catalog_model(name) for name in ALL_MODELS)
        assert is_catalog_model("unknown-model") is False


def _saved_config(path: Path, key: str, value: str) -> JsonConfigAdapter:
    config = JsonConfigAdapter(path)
    config.load()
    config.set(key, value)
    config.save()
    reloaded = JsonConfigAdapter(path)
    reloaded.load()
    return reloaded


class TestRemovedModelMigration:
    @pytest.mark.parametrize(
        ("removed_model", "replacement"), _REMOVED_MODEL_EXPECTATIONS,
    )
    def test_each_removed_model_should_migrate_to_a_surviving_model(
        self, tmp_path: Path, removed_model: str, replacement: str,
    ) -> None:
        path = tmp_path / f"{removed_model}.json"
        config = _saved_config(path, "transcription.model_size", removed_model)

        migrate_removed_models(config)

        persisted = JsonConfigAdapter(path)
        persisted.load()
        assert persisted.get("transcription.model_size") == replacement
        assert replacement in ALL_MODELS

    def test_current_model_should_remain_unchanged(self, tmp_path: Path) -> None:
        config = _saved_config(
            tmp_path / "config.json", "transcription.model_size", DEFAULT_MODEL,
        )

        migrate_removed_models(config)

        assert config.get("transcription.model_size") == DEFAULT_MODEL


class TestRemovedEngineMigration:
    @pytest.mark.parametrize("removed_engine", _REMOVED_ENGINES)
    def test_each_removed_engine_should_migrate_to_auto(
        self, tmp_path: Path, removed_engine: str,
    ) -> None:
        path = tmp_path / f"{removed_engine}.json"
        config = _saved_config(path, "transcription.engine", removed_engine)

        migrate_engine_field(config)

        persisted = JsonConfigAdapter(path)
        persisted.load()
        assert persisted.get("transcription.engine") == "auto"

    def test_missing_engine_should_migrate_to_auto(self, tmp_path: Path) -> None:
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        migrate_engine_field(config)

        assert config.get("transcription.engine") == "auto"

    def test_current_engine_should_remain_unchanged(self, tmp_path: Path) -> None:
        config = _saved_config(
            tmp_path / "config.json", "transcription.engine", "nano_parakeet",
        )

        migrate_engine_field(config)

        assert config.get("transcription.engine") == "nano_parakeet"
