from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voicetotext.core.exceptions import ConfigurationError
from voicetotext.transcription.engine_factory import (
    build_composite_model_manager,
    build_engine_registry,
    create_engine_for_active_model,
    resolve_engine_id,
)
from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.faster_whisper_adapter import FasterWhisperAdapter
from voicetotext.transcription.model_info import WHISPER_MODELS, ModelInfo
from voicetotext.transcription.transformers_whisper_adapter import (
    TransformersWhisperAdapter,
)


def _config(
    *,
    model_size: str = "base",
    device: str = "auto",
    language: str = "en",
    beam_size: int = 5,
    engine: str = "auto",
):
    config = MagicMock()
    values = {
        "transcription.model_size": model_size,
        "transcription.device": device,
        "transcription.language": language,
        "transcription.beam_size": beam_size,
        "transcription.engine": engine,
    }
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


class TestBuildCompositeModelManager:
    def test_should_return_model_manager_with_whisper_models(self, tmp_path):
        download_manager = MagicMock()

        manager = build_composite_model_manager(
            tmp_path, download_manager,
            active_engine_provider=lambda: "transformers",
        )

        names = {m.name for m in manager.list_models()}
        assert names == set(WHISPER_MODELS.keys())
