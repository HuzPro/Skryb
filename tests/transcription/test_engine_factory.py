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
from voicetotext.transcription.model_info import (
    ALL_MODELS,
    DEFAULT_MODEL,
    PARAKEET_MODELS,
    ModelInfo,
)
from voicetotext.transcription.ark_asr_adapter import ArkAsrAdapter
from voicetotext.transcription.nano_parakeet_adapter import NanoParakeetAdapter


def _config(
    *,
    model_size: str = DEFAULT_MODEL,
    device: str = "auto",
    engine: str = "auto",
) -> MagicMock:
    config = MagicMock()
    values = {
        "transcription.model_size": model_size,
        "transcription.device": device,
        "transcription.language": "en",
        "transcription.engine": engine,
    }
    config.get.side_effect = lambda key, default=None: values.get(key, default)
    return config


def _paths(models_dir: Path) -> MagicMock:
    paths = MagicMock()
    paths.models_dir = models_dir
    return paths


_GPU = HardwareProfile(
    kind="amd_rocm", device_name="AMD Radeon", gfx_arch="gfx1200",
)
_CPU = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)

_DUAL_ENGINE_MODEL = ModelInfo(
    name="dual", size_mb=1, description="test", repo_id="example/dual",
    compatible_engines=("nano_parakeet", "nemotron_streaming"),
    engine_priorities={"nano_parakeet": 100, "nemotron_streaming": 50},
)


class TestBuildEngineRegistry:
    def test_should_register_only_surviving_engines(self) -> None:
        registry = build_engine_registry()

        assert {d.engine_id for d in registry.all_descriptors()} == {
            "nano_parakeet", "nemotron_streaming", "ark_asr",
        }


class TestResolveEngineId:
    def test_should_return_explicit_engine(self) -> None:
        engine_id = resolve_engine_id(
            _config(engine="nano_parakeet"),
            PARAKEET_MODELS[DEFAULT_MODEL],
            _GPU,
            build_engine_registry(),
        )

        assert engine_id == "nano_parakeet"

    def test_should_pick_recommended_engine_in_auto_mode(self) -> None:
        engine_id = resolve_engine_id(
            _config(), PARAKEET_MODELS[DEFAULT_MODEL], _GPU,
            build_engine_registry(),
        )

        assert engine_id == "nano_parakeet"

    def test_should_prefer_installed_compatible_engine(self) -> None:
        engine_id = resolve_engine_id(
            _config(), _DUAL_ENGINE_MODEL, _GPU, build_engine_registry(),
            installed_engines=frozenset(["nemotron_streaming"]),
        )

        assert engine_id == "nemotron_streaming"

    def test_should_raise_when_no_engine_is_compatible(self) -> None:
        incompatible = ModelInfo(
            name="x", size_mb=1, description="x", compatible_engines=(),
        )

        with pytest.raises(ConfigurationError):
            resolve_engine_id(
                _config(), incompatible, _GPU, build_engine_registry(),
            )


def _fake_snapshot(models_dir: Path, repo_id: str) -> None:
    repo_dir = models_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "model.nemo").write_text("")
    refs_main = repo_dir / "refs" / "main"
    refs_main.parent.mkdir(parents=True, exist_ok=True)
    refs_main.write_text("abc123")


class TestEnginesWithModelOnDisk:
    def test_should_return_empty_when_model_is_not_cached(
        self, tmp_path: Path,
    ) -> None:
        result = engines_with_model_on_disk(
            PARAKEET_MODELS[DEFAULT_MODEL], tmp_path,
        )

        assert result == frozenset()

    def test_should_detect_loadable_parakeet_snapshot(
        self, tmp_path: Path,
    ) -> None:
        _fake_snapshot(tmp_path, "nvidia/parakeet-tdt-0.6b-v3")

        result = engines_with_model_on_disk(
            PARAKEET_MODELS[DEFAULT_MODEL], tmp_path,
        )

        assert result == frozenset(["nano_parakeet"])


class TestCreateEngineForActiveModel:
    def test_should_build_nano_parakeet_adapter(self, tmp_path: Path) -> None:
        engine = create_engine_for_active_model(
            _config(engine="nano_parakeet", device="cpu"),
            _paths(tmp_path),
            build_engine_registry(),
            _CPU,
        )

        assert isinstance(engine, NanoParakeetAdapter)
        assert engine._repo_id == "nvidia/parakeet-tdt-0.6b-v3"

    def test_should_build_ark_asr_adapter(self, tmp_path: Path) -> None:
        engine = create_engine_for_active_model(
            _config(model_size="ark-asr-3b", engine="ark_asr", device="cpu"),
            _paths(tmp_path),
            build_engine_registry(),
            _CPU,
        )

        assert isinstance(engine, ArkAsrAdapter)
        assert engine._repo_id == "Audio8/ARK-ASR-3B"

    def test_device_override_should_force_cpu(self, tmp_path: Path) -> None:
        engine = create_engine_for_active_model(
            _config(device="cuda"), _paths(tmp_path), build_engine_registry(),
            _GPU, device_override="cpu",
        )

        assert isinstance(engine, NanoParakeetAdapter)
        assert engine._device == "cpu"


class TestModelManagers:
    def test_default_manager_should_list_all_models(self, tmp_path: Path) -> None:
        manager = create_model_manager(tmp_path, MagicMock())

        assert {m.name for m in manager.list_models()} == set(ALL_MODELS)

    def test_composite_manager_should_list_all_models(
        self, tmp_path: Path,
    ) -> None:
        manager = build_composite_model_manager(
            tmp_path, MagicMock(), active_engine_provider=lambda: "nano_parakeet",
        )

        assert {m.name for m in manager.list_models()} == set(ALL_MODELS)
