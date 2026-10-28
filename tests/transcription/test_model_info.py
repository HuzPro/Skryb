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
        "underdog-bark-3.0": "85db2ad18ab41e534a222ee253c29191d77ed919",
        "underdog-bark-3.0-plus": "d4fab2f8b95681e5ad3cfb4e8c5f4e0c009b3deb",
        "underdog-bark-3.0-clean": "6d34ed8639b2b9a33e764acf06748519bbb950cf",
        "underdog-bark-3.0-pro": "4b5c51e04aa8d16dda9ecdff71bc29f7360149e4",
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


class TestBarkPlusSpeechStyles:
    def test_clean_is_the_intended_style_and_plus_the_verbatim_one(self) -> None:
        from skryb.transcription.model_info import speech_style_of

        assert speech_style_of("underdog-bark-3.0-clean") == ("underdog-bark-3.0-plus", "intended")
        assert speech_style_of("underdog-bark-3.0-plus") == ("underdog-bark-3.0-plus", "verbatim")

    def test_other_models_belong_to_no_family(self) -> None:
        from skryb.transcription.model_info import ALL_MODELS, speech_style_of

        others = set(ALL_MODELS) - {"underdog-bark-3.0-clean", "underdog-bark-3.0-plus"}
        assert all(speech_style_of(name) is None for name in others)
        assert speech_style_of("") is None
        assert speech_style_of("UNDERDOG-BARK-3.0-PLUS") is None

    def test_both_styles_and_crisperwhisper_offer_the_style_choice(self) -> None:
        from skryb.transcription.model_info import has_speech_styles

        assert has_speech_styles("underdog-bark-3.0-clean")
        assert has_speech_styles("underdog-bark-3.0-plus")
        assert has_speech_styles("crisperwhisper-2.0-turbo")
        assert not has_speech_styles("underdog-bark-3.0")
        assert not has_speech_styles("underdog-bark-3.0-pro")
        assert not has_speech_styles("phonon-2")

    def test_every_family_member_is_a_real_catalog_model(self) -> None:
        from skryb.transcription.model_info import ALL_MODELS, SPEECH_STYLE_FAMILIES

        for family, styles in SPEECH_STYLE_FAMILIES.items():
            assert family in ALL_MODELS
            assert set(styles) == {"intended", "verbatim"}
            assert all(member in ALL_MODELS for member in styles.values())


class TestWhichModelsTranscribeVerbatim:
    def test_bark_plus_is_verbatim_whatever_the_crisperwhisper_setting_says(self) -> None:
        from skryb.transcription.model_info import transcribes_verbatim

        assert transcribes_verbatim("underdog-bark-3.0-plus", "intended")
        assert transcribes_verbatim("underdog-bark-3.0-plus", "verbatim")
        assert not transcribes_verbatim("underdog-bark-3.0-clean", "verbatim")
        assert not transcribes_verbatim("underdog-bark-3.0-clean", "intended")

    def test_crisperwhisper_follows_its_mode_setting(self) -> None:
        from skryb.transcription.model_info import transcribes_verbatim

        assert transcribes_verbatim("crisperwhisper-2.0-large", "verbatim")
        assert not transcribes_verbatim("crisperwhisper-2.0-large", "intended")

    def test_a_leftover_verbatim_setting_never_turns_cleanup_off_for_other_models(self) -> None:
        from skryb.transcription.model_info import transcribes_verbatim

        for name in ("phonon-2", "parakeet-ultra", "underdog-bark-3.0", "qwen3-asr-1.7b", "", "not-a-model"):
            assert not transcribes_verbatim(name, "verbatim")
