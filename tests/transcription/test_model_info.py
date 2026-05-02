from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.transcription.model_info import (
    WHISPER_MODELS,
    ModelInfo,
    migrate_engine_field,
)


class TestModelInfoEntity:
    def test_should_store_model_attributes(self):
        info = ModelInfo(name="base", size_mb=142, description="Good balance")

        assert info.name == "base"
        assert info.size_mb == 142
        assert info.description == "Good balance"

    def test_should_format_size_in_mb(self):
        info = ModelInfo(name="small", size_mb=466, description="test")

        assert info.display_size == "466 MB"

    def test_should_format_size_in_gb_for_large_models(self):
        info = ModelInfo(name="medium", size_mb=1500, description="test")

        assert info.display_size == "1.5 GB"

    def test_should_store_explicit_repo_id(self):
        info = ModelInfo(
            name="base", size_mb=290, description="test",
            repo_id="openai/whisper-base",
        )

        assert info.repo_id == "openai/whisper-base"

    def test_repo_id_defaults_to_empty(self):
        info = ModelInfo(name="base", size_mb=142, description="test")

        assert info.repo_id == ""

    def test_compatible_engines_defaults_to_transformers_only(self):
        info = ModelInfo(name="base", size_mb=142, description="test")

        assert info.compatible_engines == ("transformers",)

    def test_should_store_explicit_compatible_engines(self):
        info = ModelInfo(
            name="base", size_mb=142, description="test",
            compatible_engines=("transformers", "faster_whisper"),
        )

        assert info.compatible_engines == ("transformers", "faster_whisper")

    def test_engine_priorities_default_to_empty_mapping(self):
        info = ModelInfo(name="base", size_mb=142, description="test")

        assert dict(info.engine_priorities) == {}

    def test_should_store_explicit_engine_priorities(self):
        info = ModelInfo(
            name="base", size_mb=142, description="test",
            engine_priorities={"faster_whisper": 100, "transformers": 50},
        )

        assert info.engine_priorities["faster_whisper"] == 100
        assert info.engine_priorities["transformers"] == 50

    def test_ct2_repo_id_defaults_to_empty(self):
        info = ModelInfo(name="base", size_mb=142, description="test")

        assert info.ct2_repo_id == ""

    def test_should_store_explicit_ct2_repo_id(self):
        info = ModelInfo(
            name="base", size_mb=142, description="test",
            ct2_repo_id="Systran/faster-whisper-base",
        )

        assert info.ct2_repo_id == "Systran/faster-whisper-base"


class TestWhisperModels:
    def test_should_contain_all_standard_models(self):
        expected = {"tiny", "base", "small", "large-v3-turbo", "large-v3"}

        assert set(WHISPER_MODELS.keys()) == expected

    def test_all_models_should_have_openai_repo_ids(self):
        for model in WHISPER_MODELS.values():
            assert model.repo_id.startswith("openai/whisper-")

    def test_all_models_should_have_systran_ct2_repo_ids(self):
        for model in WHISPER_MODELS.values():
            assert model.ct2_repo_id.startswith("Systran/faster-whisper-")

    def test_all_models_should_be_compatible_with_both_engines(self):
        for model in WHISPER_MODELS.values():
            assert "transformers" in model.compatible_engines
            assert "faster_whisper" in model.compatible_engines

    def test_faster_whisper_should_outrank_transformers_for_every_model(self):
        for model in WHISPER_MODELS.values():
            assert model.engine_priorities["faster_whisper"] > model.engine_priorities["transformers"]


class TestMigrateEngineField:
    def test_should_set_engine_to_auto_when_missing(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        migrate_engine_field(config)

        assert config.get("transcription.engine") == "auto"

    def test_should_leave_engine_unchanged_when_explicitly_set(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()
        config.set("transcription.engine", "faster_whisper")

        migrate_engine_field(config)

        assert config.get("transcription.engine") == "faster_whisper"

    def test_should_persist_migration_to_disk(self, tmp_path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()

        migrate_engine_field(config)

        reloaded = JsonConfigAdapter(path)
        reloaded.load()
        assert reloaded.get("transcription.engine") == "auto"


