import pytest

from voicetotext.core.exceptions import ConfigurationError
from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.transcription.model_info import (
    ALL_MODELS,
    GRANITE_MODELS,
    NEMOTRON_STREAMING_MODELS,
    PARAKEET_MODELS,
    WHISPER_MODELS,
    ModelInfo,
    allows_device_selection,
    get_model_info,
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

    def test_ct2_repo_ids_are_official_systran_when_present(self):
        # A CT2 repo id, when set, must point at the official Systran
        # mirror. large-v3-turbo deliberately has none (Systran never
        # published a turbo CT2 repo — the old id 401'd), so it is skipped.
        for model in WHISPER_MODELS.values():
            if model.ct2_repo_id:
                assert model.ct2_repo_id.startswith("Systran/faster-whisper-")

    def test_faster_whisper_compatibility_matches_ct2_repo_presence(self):
        # Every Whisper model runs on transformers; faster-whisper is
        # offered only when there is a real CT2 repo to load from.
        for model in WHISPER_MODELS.values():
            assert "transformers" in model.compatible_engines
            assert ("faster_whisper" in model.compatible_engines) == bool(
                model.ct2_repo_id
            )

    def test_large_v3_turbo_is_transformers_only(self):
        # Systran has no faster-whisper-large-v3-turbo CT2 repo and the only
        # working mirror is community-hosted, so turbo runs through the
        # transformers engine alone (no ct2_repo_id, no faster_whisper).
        turbo = WHISPER_MODELS["large-v3-turbo"]

        assert turbo.ct2_repo_id == ""
        assert turbo.compatible_engines == ("transformers",)

    def test_faster_whisper_should_outrank_transformers_when_both_offered(self):
        for model in WHISPER_MODELS.values():
            if "faster_whisper" in model.compatible_engines:
                assert (
                    model.engine_priorities["faster_whisper"]
                    > model.engine_priorities["transformers"]
                )


class TestParakeetModels:
    def test_should_contain_tdt_v3(self):
        expected = {"parakeet-tdt-0.6b-v3"}

        assert set(PARAKEET_MODELS.keys()) == expected

    def test_all_models_should_have_nvidia_repo_ids(self):
        for model in PARAKEET_MODELS.values():
            assert model.repo_id.startswith("nvidia/parakeet-tdt-")

    def test_all_models_should_be_compatible_with_nano_parakeet_only(self):
        for model in PARAKEET_MODELS.values():
            assert model.compatible_engines == ("nano_parakeet",)

    def test_all_models_should_have_no_ct2_repo_id(self):
        for model in PARAKEET_MODELS.values():
            assert model.ct2_repo_id == ""


class TestAllModels:
    def test_should_contain_all_engine_specific_model_dicts(self):
        expected = (
            set(WHISPER_MODELS.keys())
            | set(PARAKEET_MODELS.keys())
            | set(NEMOTRON_STREAMING_MODELS.keys())
            | set(GRANITE_MODELS.keys())
        )

        assert set(ALL_MODELS.keys()) == expected

    def test_whisper_entries_should_match_whisper_models(self):
        for name, info in WHISPER_MODELS.items():
            assert ALL_MODELS[name] is info

    def test_parakeet_entries_should_match_parakeet_models(self):
        for name, info in PARAKEET_MODELS.items():
            assert ALL_MODELS[name] is info

    def test_granite_entries_should_match_granite_models(self):
        for name, info in GRANITE_MODELS.items():
            assert ALL_MODELS[name] is info


class TestNemotronStreamingModels:
    def test_should_contain_only_the_streaming_model(self):
        assert set(NEMOTRON_STREAMING_MODELS.keys()) == {
            "nemotron-speech-streaming-en-0.6b",
        }

    def test_streaming_model_is_nemotron_streaming_only(self):
        info = get_model_info("nemotron-speech-streaming-en-0.6b")

        assert info.compatible_engines == ("nemotron_streaming",)


class TestGraniteModels:
    def test_all_models_should_have_repo_id(self):
        for model in GRANITE_MODELS.values():
            assert model.repo_id != ""

    def test_all_models_should_be_compatible_with_granite_speech_only(self):
        for model in GRANITE_MODELS.values():
            assert model.compatible_engines == ("granite_speech",)

    def test_all_models_should_have_no_ct2_repo_id(self):
        for model in GRANITE_MODELS.values():
            assert model.ct2_repo_id == ""


class TestGetModelInfo:
    def test_should_return_whisper_model_by_name(self):
        info = get_model_info("base")

        assert info.name == "base"

    def test_should_return_parakeet_model_by_name(self):
        info = get_model_info("parakeet-tdt-0.6b-v3")

        assert info.name == "parakeet-tdt-0.6b-v3"

    def test_should_raise_configuration_error_for_unknown_model(self):
        with pytest.raises(ConfigurationError, match="unknown-model"):
            get_model_info("unknown-model")


class TestAllowsDeviceSelection:
    def test_should_allow_device_selection_for_tiny(self):
        assert allows_device_selection("tiny") is True

    def test_should_allow_device_selection_for_base(self):
        assert allows_device_selection("base") is True

    def test_should_disallow_device_selection_for_small(self):
        assert allows_device_selection("small") is False

    def test_should_disallow_device_selection_for_large_v3_turbo(self):
        assert allows_device_selection("large-v3-turbo") is False

    def test_should_disallow_device_selection_for_unknown_model(self):
        assert allows_device_selection("unknown-model") is False


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


