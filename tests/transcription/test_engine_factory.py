from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voicetotext.core.exceptions import ConfigurationError
from voicetotext.transcription.engine_factory import (
    build_composite_model_manager,
    build_engine_registry,
    create_engine_for_active_model,
    create_model_manager,
    engines_with_model_on_disk,
    resolve_engine_id,
)
from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.faster_whisper_adapter import FasterWhisperAdapter
from voicetotext.transcription.model_info import (
    ALL_MODELS,
    PARAKEET_MODELS,
    WHISPER_MODELS,
    ModelInfo,
)
from voicetotext.transcription.nano_parakeet_adapter import NanoParakeetAdapter
from voicetotext.transcription.transformers_whisper_adapter import (
    TransformersWhisperAdapter,
)


def _config(
    *,
    model_size: str = "base",
    device: str = "auto",
    language: str = "en",
    beam_size: int = 5,
    beam_size_set: bool = True,
    engine: str = "auto",
):
    config = MagicMock()
    values = {
        "transcription.model_size": model_size,
        "transcription.device": device,
        "transcription.language": language,
        "transcription.engine": engine,
    }
    if beam_size_set:
        values["transcription.beam_size"] = beam_size
    config.get.side_effect = lambda key, default=None: values.get(key, default)
    return config


def _paths(models_dir: Path) -> object:
    paths = MagicMock()
    paths.models_dir = models_dir
    return paths


_ROCM_PROFILE = HardwareProfile(
    kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200",
)
_CPU_PROFILE = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)


class TestBuildEngineRegistry:
    def test_should_register_transformers_descriptor(self):
        registry = build_engine_registry()

        ids = {d.engine_id for d in registry.all_descriptors()}
        assert "transformers" in ids

    def test_should_register_faster_whisper_descriptor(self):
        registry = build_engine_registry()

        ids = {d.engine_id for d in registry.all_descriptors()}
        assert "faster_whisper" in ids


    def test_should_register_nano_parakeet_descriptor(self):
        registry = build_engine_registry()

        ids = {d.engine_id for d in registry.all_descriptors()}
        assert "nano_parakeet" in ids

    def test_should_register_granite_speech_descriptor(self):
        registry = build_engine_registry()

        ids = {d.engine_id for d in registry.all_descriptors()}
        assert "granite_speech" in ids


class TestResolveEngineId:
    def test_should_return_explicit_engine_when_config_specifies_one(self):
        config = _config(engine="transformers")
        registry = build_engine_registry()

        engine_id = resolve_engine_id(
            config, WHISPER_MODELS["base"], _ROCM_PROFILE, registry,
        )

        assert engine_id == "transformers"

    def test_should_pick_recommended_engine_when_config_engine_is_auto(self):
        config = _config(engine="auto")
        registry = build_engine_registry()

        engine_id = resolve_engine_id(
            config, WHISPER_MODELS["base"], _ROCM_PROFILE, registry,
        )

        assert engine_id == "faster_whisper"

    def test_should_raise_when_no_engine_compatible(self):
        config = _config(engine="auto")
        registry = build_engine_registry()
        incompatible = ModelInfo(
            name="x", size_mb=1, description="x",
            compatible_engines=(),  # nothing matches
        )

        with pytest.raises(ConfigurationError):
            resolve_engine_id(config, incompatible, _ROCM_PROFILE, registry)

    def test_should_prefer_installed_engine_over_higher_ranked_uninstalled(self):
        config = _config(engine="auto")
        registry = build_engine_registry()

        engine_id = resolve_engine_id(
            config, WHISPER_MODELS["base"], _ROCM_PROFILE, registry,
            installed_engines=frozenset(["transformers"]),
        )

        assert engine_id == "transformers"

    def test_should_keep_top_ranked_when_no_engines_installed(self):
        config = _config(engine="auto")
        registry = build_engine_registry()

        engine_id = resolve_engine_id(
            config, WHISPER_MODELS["base"], _ROCM_PROFILE, registry,
            installed_engines=frozenset(),
        )

        assert engine_id == "faster_whisper"

    def test_should_ignore_installed_engines_when_engine_is_explicit(self):
        config = _config(engine="faster_whisper")
        registry = build_engine_registry()

        engine_id = resolve_engine_id(
            config, WHISPER_MODELS["base"], _ROCM_PROFILE, registry,
            installed_engines=frozenset(["transformers"]),
        )

        assert engine_id == "faster_whisper"


def _fake_snapshot(models_dir: Path, repo_id: str) -> None:
    """Lay down a complete, loadable cache entry (with ``refs/main``)."""
    repo_dir = models_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_text("")
    refs_main = repo_dir / "refs" / "main"
    refs_main.parent.mkdir(parents=True, exist_ok=True)
    refs_main.write_text("abc123")


def _partial_snapshot(models_dir: Path, repo_id: str) -> None:
    """Lay down snapshot files but NO ``refs/main`` (a cancelled download)."""
    snapshot = (
        models_dir / f"models--{repo_id.replace('/', '--')}"
        / "snapshots" / "abc123"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_text("")


class TestEnginesWithModelOnDisk:
    def test_should_return_empty_when_no_models_cached(self, tmp_path):
        result = engines_with_model_on_disk(WHISPER_MODELS["base"], tmp_path)

        assert result == frozenset()

    def test_should_detect_transformers_model(self, tmp_path):
        _fake_snapshot(tmp_path, "openai/whisper-base")

        result = engines_with_model_on_disk(WHISPER_MODELS["base"], tmp_path)

        assert result == frozenset(["transformers"])

    def test_should_detect_faster_whisper_model(self, tmp_path):
        _fake_snapshot(tmp_path, "Systran/faster-whisper-base")

        result = engines_with_model_on_disk(WHISPER_MODELS["base"], tmp_path)

        assert result == frozenset(["faster_whisper"])

    def test_should_detect_both_engines(self, tmp_path):
        _fake_snapshot(tmp_path, "openai/whisper-base")
        _fake_snapshot(tmp_path, "Systran/faster-whisper-base")

        result = engines_with_model_on_disk(WHISPER_MODELS["base"], tmp_path)

        assert result == frozenset(["transformers", "faster_whisper"])

    def test_should_ignore_partial_snapshot_without_refs_main(self, tmp_path):
        # A cancelled download leaves snapshot files but no refs/main. The
        # loader (local_files_only) cannot open it, so it must not count as
        # an engine-on-disk and bias selection toward a broken cache.
        _partial_snapshot(tmp_path, "Systran/faster-whisper-base")

        result = engines_with_model_on_disk(WHISPER_MODELS["base"], tmp_path)

        assert result == frozenset()


class TestCreateEngineForActiveModel:
    def test_should_build_transformers_adapter_when_engine_is_transformers(
        self, tmp_path,
    ):
        config = _config(engine="transformers", device="cpu")
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert isinstance(engine, TransformersWhisperAdapter)

    def test_should_build_faster_whisper_adapter_when_engine_is_faster_whisper(
        self, tmp_path,
    ):
        config = _config(engine="faster_whisper", device="cpu")
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert isinstance(engine, FasterWhisperAdapter)

    def test_device_override_should_force_cpu(self, tmp_path):
        config = _config(engine="transformers", device="cuda")
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _ROCM_PROFILE,
            device_override="cpu",
        )

        assert engine._device == "cpu"

    def test_should_route_ct2_repo_id_to_faster_whisper_adapter(self, tmp_path):
        config = _config(engine="faster_whisper", device="cpu")
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert engine._repo_id == "Systran/faster-whisper-base"

    def test_should_prefer_installed_engine_in_auto_mode(self, tmp_path):
        _fake_snapshot(tmp_path, "openai/whisper-base")
        config = _config(engine="auto", device="cpu")
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert isinstance(engine, TransformersWhisperAdapter)

    def test_should_build_nano_parakeet_adapter_for_parakeet_model(
        self, tmp_path,
    ):
        config = _config(
            model_size="parakeet-tdt-0.6b-v3",
            engine="nano_parakeet",
            device="cpu",
        )
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert isinstance(engine, NanoParakeetAdapter)

    def test_should_route_nvidia_repo_id_to_nano_parakeet_adapter(
        self, tmp_path,
    ):
        config = _config(
            model_size="parakeet-tdt-0.6b-v3",
            engine="nano_parakeet",
            device="cpu",
        )
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert engine._repo_id == "nvidia/parakeet-tdt-0.6b-v3"

    def test_should_auto_select_nano_parakeet_for_parakeet_model(
        self, tmp_path,
    ):
        config = _config(
            model_size="parakeet-tdt-0.6b-v3",
            engine="auto",
            device="cpu",
        )
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert isinstance(engine, NanoParakeetAdapter)

    def test_should_build_granite_speech_adapter_for_granite_model(
        self, tmp_path,
    ):
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        config = _config(
            model_size="granite-speech-4.1-2b",
            engine="granite_speech",
            device="cpu",
        )
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert isinstance(engine, GraniteSpeechAdapter)
        assert engine._repo_id == "ibm-granite/granite-speech-4.1-2b"

    def test_should_auto_select_granite_speech_for_granite_model(
        self, tmp_path,
    ):
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        config = _config(
            model_size="granite-speech-4.1-2b",
            engine="auto",
            device="cpu",
        )
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert isinstance(engine, GraniteSpeechAdapter)

    def test_should_pass_configured_beam_size_to_granite_adapter(
        self, tmp_path,
    ):
        # Users can trade Granite accuracy for ~Nx speed by setting
        # ``transcription.beam_size`` below the IBM-recommended 4.
        # The factory must forward whatever the config holds rather
        # than silently keeping the adapter on its default.
        config = _config(
            model_size="granite-speech-4.1-2b",
            engine="granite_speech",
            device="cpu",
            beam_size=2,
        )
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert engine._num_beams == 2

    def test_should_default_granite_to_ibm_recipe_when_beam_size_unset(
        self, tmp_path,
    ):
        # IBM's tech report recommends ``num_beams=4`` for Granite
        # Speech. When the user has not explicitly set
        # ``transcription.beam_size``, the factory must defer to the
        # per-engine recommendation rather than the whisper default
        # of 5 — otherwise every default-config user pays a ~25%
        # decoder latency tax on Granite.
        config = _config(
            model_size="granite-speech-4.1-2b",
            engine="granite_speech",
            device="cpu",
            beam_size_set=False,
        )
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert engine._num_beams == 4

    def test_should_default_whisper_to_five_when_beam_size_unset(
        self, tmp_path,
    ):
        # Regression guard for the per-engine default lookup: changing
        # the unset-default routing must not lower whisper beam from 5.
        config = _config(
            engine="transformers",
            device="cpu",
            beam_size_set=False,
        )
        registry = build_engine_registry()

        engine = create_engine_for_active_model(
            config, _paths(tmp_path), registry, _CPU_PROFILE,
        )

        assert engine._beam_size == 5


class TestCreateModelManager:
    def test_should_return_model_manager_listing_all_models(self, tmp_path):
        download_manager = MagicMock()

        manager = create_model_manager(tmp_path, download_manager)

        names = {m.name for m in manager.list_models()}
        assert names == set(ALL_MODELS.keys())

    def test_should_report_model_not_installed_on_empty_directory(self, tmp_path):
        download_manager = MagicMock()

        manager = create_model_manager(tmp_path, download_manager)

        assert manager.is_installed("base") is False


class TestBuildCompositeModelManager:
    def test_should_return_model_manager_with_all_models(self, tmp_path):
        download_manager = MagicMock()

        manager = build_composite_model_manager(
            tmp_path, download_manager,
            active_engine_provider=lambda: "transformers",
        )

        names = {m.name for m in manager.list_models()}
        assert names == set(ALL_MODELS.keys())

    def test_should_not_register_turbo_in_faster_whisper_child(self, tmp_path):
        # turbo has no ct2_repo_id, so _ct2_models excludes it: the
        # faster-whisper child must never advertise a turbo download (the
        # old Systran turbo repo 401'd).
        manager = build_composite_model_manager(
            tmp_path, MagicMock(), active_engine_provider=lambda: "faster_whisper",
        )

        # The aggregate catalogue still lists turbo (via the transformers
        # child), but the faster-whisper child does not own it.
        assert manager.has_model("large-v3-turbo") is True
        fw_child = manager._by_engine["faster_whisper"]
        assert fw_child.has_model("large-v3-turbo") is False


class TestInstallStateMatchesLoadableEngine:
    """Regression for the large-v3 / large-v3-turbo Download-vs-Delete
    confusion: the install state must reflect the engine that will actually
    load the model, not a top-ranked engine whose cache is empty.
    """

    def test_should_report_turbo_installed_from_transformers_snapshot(
        self, tmp_path,
    ):
        # turbo lives only under the openai (transformers) cache. Even when
        # faster_whisper is the active/top-ranked engine, the composite must
        # report turbo installed because transformers is what loads it.
        _fake_snapshot(tmp_path, "openai/whisper-large-v3-turbo")
        manager = build_composite_model_manager(
            tmp_path, MagicMock(),
            active_engine_provider=lambda: "faster_whisper",
        )

        assert manager.is_installed("large-v3-turbo") is True
