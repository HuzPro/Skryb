from pathlib import Path
from typing import ClassVar

import pytest

from skryb.core.exceptions import ConfigurationError
from skryb.core.json_config_adapter import JsonConfigAdapter
from skryb.transcription.model_info import (
    ALL_MODELS,
    CRISPERWHISPER_MODELS,
    DEFAULT_MODEL,
    NEMOTRON_STREAMING_MODELS,
    PARAKEET_MODELS,
    PHONON2_MODELS,
    QWEN3_ASR_MODELS,
    ModelInfo,
    default_model_for,
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
    ("granite-speech-5.0-470m-turboctc", DEFAULT_MODEL),
    ("ark-asr-3b", DEFAULT_MODEL),
    ("parakeet-tdt-0.6b-v2", DEFAULT_MODEL),
    ("nemotron-streaming-compare", "nemotron-speech-streaming-en-0.6b"),
    ("underdog-bark-3.0", "phonon-2"),
    ("underdog-bark-3.0-plus", "phonon-2"),
    ("underdog-bark-3.0-clean", "phonon-2"),
    ("underdog-bark-3.0-pro", "phonon-2"),
)

_REMOVED_ENGINES = (
    "ark_asr",
    "faster_whisper",
    "transformers",
    "granite_speech",
    "granite_turboctc",
    "qwen3_asr",
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

    def test_revision_should_default_to_tracking_main(self) -> None:
        info = ModelInfo(name="example", size_mb=1, description="Example")

        assert info.revision == ""

    def test_license_should_default_to_ungated(self) -> None:
        info = ModelInfo(name="example", size_mb=1, description="Example")

        assert info.license_id == ""
        assert info.license_url == ""


class TestPinnedRevisions:
    """CrisperWhisper ships a licence keyed to each exact commit, so both resolve one audited
    commit instead of whatever ``main`` points at today.
    """

    _EXPECTED_PINS: ClassVar[dict[str, str]] = {
        "crisperwhisper-2.0-turbo": "de0369c8a68025b7f6e86387b6eb5a3b369787c8",
        "crisperwhisper-2.0-large": "f4334f6e8193f2691212d49b20fa12d370e13896",
        "parakeet-ultra": "73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b",
        "phonon-2": "fb09063c9dc8e130d97b3a2191ed5aed6b71bb31",
        "qwen3-asr-1.7b": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    }

    @pytest.mark.parametrize(
        ("model_name", "revision"), sorted(_EXPECTED_PINS.items()),
    )
    def test_each_pinned_model_should_carry_its_audited_commit(
        self, model_name: str, revision: str,
    ) -> None:
        assert get_model_info(model_name).revision == revision

    def test_every_other_model_should_track_main(self) -> None:
        unpinned = {
            name for name, info in ALL_MODELS.items() if not info.revision
        }

        assert unpinned == set(ALL_MODELS) - set(self._EXPECTED_PINS)


class TestGatedLicences:
    """Both CrisperWhisper variants ship the same licence bytes, so they
    share one ``license_id`` and one acceptance. Their ``license_url``
    must point at the pinned commit, never at a branch that can be
    edited under the acceptance.
    """

    def test_both_crisperwhisper_variants_should_share_one_licence(self) -> None:
        license_ids = {
            info.license_id for info in CRISPERWHISPER_MODELS.values()
        }

        assert license_ids == {"crisperwhisper-nc"}

    @pytest.mark.parametrize("model_name", sorted(CRISPERWHISPER_MODELS))
    def test_licence_url_should_name_the_pinned_commit(
        self, model_name: str,
    ) -> None:
        info = get_model_info(model_name)

        assert info.revision in info.license_url
        assert "/main/" not in info.license_url

    def test_every_other_model_should_be_ungated(self) -> None:
        gated = {name for name, info in ALL_MODELS.items() if info.license_id}

        assert gated == set(CRISPERWHISPER_MODELS)


class TestModelCatalog:
    def test_should_contain_only_surviving_model_lineups(self) -> None:
        assert set(ALL_MODELS) == (
            set(PARAKEET_MODELS)
            | set(PHONON2_MODELS)
            | set(NEMOTRON_STREAMING_MODELS)
            | set(CRISPERWHISPER_MODELS)
            | set(QWEN3_ASR_MODELS)
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

    def test_crisperwhisper_models_should_disclose_restrictive_license(
        self,
    ) -> None:
        assert CRISPERWHISPER_MODELS
        for model in CRISPERWHISPER_MODELS.values():
            assert "non-commercial research license" in model.description
            assert model.compatible_engines[0] == "crisperwhisper"
            assert set(model.compatible_engines) <= {
                "crisperwhisper", "crisperwhisper_ct2",
            }

    def test_crisperwhisper_should_not_replace_default_model(self) -> None:
        assert DEFAULT_MODEL not in CRISPERWHISPER_MODELS


def _saved_config(path: Path, key: str, value: str) -> JsonConfigAdapter:
    config = JsonConfigAdapter(path)
    config.load()
    config.set(key, value)
    config.save()
    reloaded = JsonConfigAdapter(path)
    reloaded.load()
    return reloaded


def _loaded_config(path: Path) -> JsonConfigAdapter:
    config = JsonConfigAdapter(path)
    config.load()
    return config


class TestRemovedModelMigration:
    @pytest.mark.parametrize(
        "bark", ["underdog-bark-3.0", "underdog-bark-3.0-plus", "underdog-bark-3.0-clean", "underdog-bark-3.0-pro"],
    )
    def test_a_bark_user_in_live_mode_moves_to_a_model_that_streams(self, tmp_path: Path, bark: str) -> None:
        path = tmp_path / "config.json"
        config = _saved_config(path, "transcription.model_size", bark)
        config.set("transcription.mode", "streaming")

        migrate_removed_models(config)

        persisted = JsonConfigAdapter(path)
        persisted.load()
        assert persisted.get("transcription.model_size") == DEFAULT_MODEL
        assert persisted.get("transcription.mode") == "streaming"

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

    def test_a_current_model_keeps_the_engine_it_was_pinned_to(self, tmp_path: Path) -> None:
        config = _saved_config(tmp_path / "config.json", "transcription.model_size", "parakeet-ultra")
        config.set("transcription.engine", "openvino_parakeet")

        migrate_removed_models(config)

        assert config.get("transcription.engine") == "openvino_parakeet"


class TestABarkConfigMovesToPhonon2:
    @staticmethod
    def _bark_config(path: Path, model: str) -> JsonConfigAdapter:
        config = JsonConfigAdapter(path)
        config.load()
        for key, value in {
            "transcription.model_size": model,
            "transcription.engine": "openvino_parakeet",
            "transcription.crisperwhisper_mode": "verbatim",
            "transcription.language": "en",
            "audio.input_device": "USB Microphone (Røde NT-USB Mini)",
            "hotkey.double_tap_locks_hold": False,
            "history.retention_days": 0,
            "dictionary.learn_words": True,
            "output.paste_method_overrides": {
                "WINWORD.EXE": "ctrl_shift_v", "Code - Insiders.exe": "type", "Zoë's Notes.exe": "type", "": "type",
            },
        }.items():
            config.set(key, value)
        config.save()
        return _loaded_config(path)

    @pytest.mark.parametrize(
        "bark", ["underdog-bark-3.0", "underdog-bark-3.0-plus", "underdog-bark-3.0-clean", "underdog-bark-3.0-pro"],
    )
    def test_every_bark_model_lands_on_phonon_2_with_the_engine_back_on_auto(
        self, tmp_path: Path, bark: str,
    ) -> None:
        path = tmp_path / "config.json"
        expected = self._bark_config(path, bark)
        expected.set("transcription.model_size", "phonon-2")
        expected.set("transcription.engine", "auto")

        migrate_removed_models(_loaded_config(path))

        assert _loaded_config(path).get_all() == expected.get_all()

    def test_the_migrated_config_resolves_to_phonon_2s_cpu_engine(self, tmp_path: Path) -> None:
        from skryb.transcription.engine_factory import build_engine_registry, selectable_engine_ids

        config = self._bark_config(tmp_path / "config.json", "underdog-bark-3.0-plus")

        migrate_removed_models(config)
        migrate_engine_field(config)

        assert selectable_engine_ids(config, build_engine_registry()) == ("phonon2_cpu",)

    def test_a_second_launch_leaves_the_migrated_config_alone(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        migrate_removed_models(self._bark_config(path, "underdog-bark-3.0-pro"))
        config = _loaded_config(path)
        config.set("transcription.engine", "phonon2_cpu")
        config.save()

        migrate_removed_models(_loaded_config(path))

        assert _loaded_config(path).get("transcription.engine") == "phonon2_cpu"

    def test_names_that_only_look_like_bark_are_left_alone(self, tmp_path: Path) -> None:
        for name in ("UNDERDOG-BARK-3.0", "underdog-bark-3.0 ", "underdog-bark-4.0", "bark"):
            config = _saved_config(tmp_path / "config.json", "transcription.model_size", name)

            migrate_removed_models(config)

            assert config.get("transcription.model_size") == name


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


class TestDefaultModelForHardware:
    def test_a_machine_without_a_gpu_starts_on_phonon_2(self) -> None:
        assert default_model_for("cpu") == "phonon-2"

    @pytest.mark.parametrize("gpu", ["nvidia_cuda", "amd_rocm"])
    def test_a_machine_with_a_gpu_keeps_parakeet(self, gpu: str) -> None:
        assert default_model_for(gpu) == DEFAULT_MODEL

    def test_phonon_2_is_ungated_english_only_and_cpu_engine_only(self) -> None:
        info = get_model_info("phonon-2")

        assert info.license_id == ""
        assert "English only" in info.description
        assert info.compatible_engines == ("phonon2_cpu",)
        assert info.display_size == "164 MB"


class TestWhichModelsTranscribeVerbatim:
    def test_crisperwhisper_follows_its_mode_setting(self) -> None:
        from skryb.transcription.model_info import transcribes_verbatim

        assert transcribes_verbatim("crisperwhisper-2.0-large", "verbatim")
        assert not transcribes_verbatim("crisperwhisper-2.0-large", "intended")

    def test_a_leftover_verbatim_setting_never_turns_cleanup_off_for_other_models(self) -> None:
        from skryb.transcription.model_info import transcribes_verbatim

        for name in ("phonon-2", "parakeet-ultra", "underdog-bark-3.0-plus", "qwen3-asr-1.7b", "", "not-a-model"):
            assert not transcribes_verbatim(name, "verbatim")


class TestModelSizes:
    def test_sizes_round_to_gigabytes_from_a_thousand_megabytes(self) -> None:
        from skryb.transcription.model_info import format_size_mb

        assert [format_size_mb(size) for size in (0, 164, 999, 1000, 1254, 4100)] == [
            "0 MB", "164 MB", "999 MB", "1.0 GB", "1.3 GB", "4.1 GB",
        ]
