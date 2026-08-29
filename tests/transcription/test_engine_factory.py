from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.fakes.config import FakeConfig
from tests.fakes.licensing import AllowAllLicensePolicy
from voicetotext.core.exceptions import ConfigurationError, LicenseNotAcceptedError
from voicetotext.transcription.license_policy import LicensePolicy
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
from voicetotext.transcription.qwen3_asr_adapter import Qwen3AsrAdapter
from voicetotext.transcription.crisperwhisper_adapter import (
    CrisperWhisperAdapter,
)


def _config(
    *,
    model_size: str = DEFAULT_MODEL,
    device: str = "auto",
    engine: str = "auto",
    crisperwhisper_mode: str = "intended",
) -> MagicMock:
    config = MagicMock()
    values = {
        "transcription.model_size": model_size,
        "transcription.device": device,
        "transcription.language": "en",
        "transcription.engine": engine,
        "transcription.crisperwhisper_mode": crisperwhisper_mode,
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

_PINNED_SHA = "1e28271b79edc97635783bea65abc89195a09ed3"
_PINNED_MODEL = ModelInfo(
    name="pinned", size_mb=1, description="test", repo_id="example/pinned",
    revision=_PINNED_SHA, compatible_engines=("nano_parakeet",),
)


class TestBuildEngineRegistry:
    def test_should_register_only_surviving_engines(self) -> None:
        registry = build_engine_registry()

        assert {d.engine_id for d in registry.all_descriptors()} == {
            "nano_parakeet", "nemotron_streaming", "ark_asr", "qwen3_asr",
            "crisperwhisper",
        }

    def test_crisperwhisper_should_have_lower_base_priority(self) -> None:
        by_id = {d.engine_id: d for d in build_engine_registry().all_descriptors()}

        assert by_id["crisperwhisper"].base_priority < (
            by_id["nano_parakeet"].base_priority
        )


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


def _fake_snapshot(models_dir: Path, repo_id: str, sha: str = "abc123") -> None:
    repo_dir = models_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "model.nemo").write_text("")
    refs_main = repo_dir / "refs" / "main"
    refs_main.parent.mkdir(parents=True, exist_ok=True)
    refs_main.write_text(sha)


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

    def test_should_ignore_a_cache_at_a_different_commit_than_the_pin(
        self, tmp_path: Path,
    ) -> None:
        _fake_snapshot(tmp_path, _PINNED_MODEL.repo_id, sha="stale")

        result = engines_with_model_on_disk(_PINNED_MODEL, tmp_path)

        assert result == frozenset()

    def test_should_detect_a_cache_at_the_pinned_commit(
        self, tmp_path: Path,
    ) -> None:
        _fake_snapshot(tmp_path, _PINNED_MODEL.repo_id, sha=_PINNED_SHA)

        result = engines_with_model_on_disk(_PINNED_MODEL, tmp_path)

        assert result == frozenset(["nano_parakeet"])


class TestCreateEngineForActiveModel:
    def test_should_build_nano_parakeet_adapter(self, tmp_path: Path) -> None:
        engine = create_engine_for_active_model(
            _config(engine="nano_parakeet", device="cpu"),
            _paths(tmp_path),
            build_engine_registry(),
            _CPU,
            license_policy=AllowAllLicensePolicy(),
        )

        assert isinstance(engine, NanoParakeetAdapter)
        assert engine._repo_id == "nvidia/parakeet-tdt-0.6b-v3"

    def test_should_build_ark_asr_adapter_pinned_to_the_catalog_revision(
        self, tmp_path: Path,
    ) -> None:
        engine = create_engine_for_active_model(
            _config(model_size="ark-asr-3b", engine="ark_asr", device="cpu"),
            _paths(tmp_path),
            build_engine_registry(),
            _CPU,
            license_policy=AllowAllLicensePolicy(),
        )

        assert isinstance(engine, ArkAsrAdapter)
        assert engine._repo_id == "Audio8/ARK-ASR-3B"
        assert engine._revision == ALL_MODELS["ark-asr-3b"].revision

    def test_should_build_qwen3_asr_adapter(self, tmp_path: Path) -> None:
        engine = create_engine_for_active_model(
            _config(
                model_size="qwen3-asr-1.7b",
                engine="qwen3_asr",
                device="cpu",
            ),
            _paths(tmp_path),
            build_engine_registry(),
            _CPU,
            license_policy=AllowAllLicensePolicy(),
        )

        assert isinstance(engine, Qwen3AsrAdapter)
        assert engine._repo_id == "Qwen/Qwen3-ASR-1.7B-hf"
        assert engine._language == "en"

    def test_device_override_should_force_cpu(self, tmp_path: Path) -> None:
        engine = create_engine_for_active_model(
            _config(device="cuda"), _paths(tmp_path), build_engine_registry(),
            _GPU, license_policy=AllowAllLicensePolicy(), device_override="cpu",
        )

        assert isinstance(engine, NanoParakeetAdapter)
        assert engine._device == "cpu"

    def test_should_build_crisperwhisper_with_default_intended_mode(
        self, tmp_path: Path,
    ) -> None:
        engine = create_engine_for_active_model(
            _config(
                model_size="crisperwhisper-2.0-turbo",
                engine="crisperwhisper",
                device="cpu",
            ),
            _paths(tmp_path),
            build_engine_registry(),
            _CPU,
            license_policy=AllowAllLicensePolicy(),
        )

        assert isinstance(engine, CrisperWhisperAdapter)
        assert engine._repo_id == "nyralabs/CrisperWhisper2.0_turbo"
        assert engine._mode == "intended"
        assert engine._revision == ALL_MODELS["crisperwhisper-2.0-turbo"].revision

    def test_should_build_crisperwhisper_large_at_its_own_pin(
        self, tmp_path: Path,
    ) -> None:
        engine = create_engine_for_active_model(
            _config(
                model_size="crisperwhisper-2.0-large",
                engine="crisperwhisper",
                device="cpu",
            ),
            _paths(tmp_path),
            build_engine_registry(),
            _CPU,
            license_policy=AllowAllLicensePolicy(),
        )

        assert isinstance(engine, CrisperWhisperAdapter)
        assert engine._revision == ALL_MODELS["crisperwhisper-2.0-large"].revision

    def test_should_pass_validated_verbatim_mode(self, tmp_path: Path) -> None:
        engine = create_engine_for_active_model(
            _config(
                model_size="crisperwhisper-2.0-large",
                engine="crisperwhisper",
                crisperwhisper_mode="verbatim",
            ),
            _paths(tmp_path),
            build_engine_registry(),
            _GPU,
            license_policy=AllowAllLicensePolicy(),
        )

        assert isinstance(engine, CrisperWhisperAdapter)
        assert engine._mode == "verbatim"

    def test_should_reject_invalid_crisperwhisper_mode(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ConfigurationError, match="crisperwhisper_mode"):
            create_engine_for_active_model(
                _config(
                    model_size="crisperwhisper-2.0-turbo",
                    engine="crisperwhisper",
                    crisperwhisper_mode="raw",
                ),
                _paths(tmp_path),
                build_engine_registry(),
                _GPU,
                license_policy=AllowAllLicensePolicy(),
            )


class TestLicenceGatedEngineConstruction:
    """Engine construction is one of the paths the gate covers: an
    unaccepted licence must stop the adapter from being built at all, not
    only stop the download.
    """

    def test_should_refuse_to_build_an_engine_for_an_unaccepted_licence(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(
            LicenseNotAcceptedError, match="crisperwhisper-2.0-turbo",
        ):
            create_engine_for_active_model(
                _config(
                    model_size="crisperwhisper-2.0-turbo",
                    engine="crisperwhisper",
                    device="cpu",
                ),
                _paths(tmp_path),
                build_engine_registry(),
                _CPU,
                license_policy=LicensePolicy(FakeConfig()),
            )

    def test_should_build_the_engine_once_the_licence_is_accepted(
        self, tmp_path: Path,
    ) -> None:
        config = FakeConfig()
        policy = LicensePolicy(config)
        policy.accept(ALL_MODELS["crisperwhisper-2.0-turbo"])

        engine = create_engine_for_active_model(
            _config(
                model_size="crisperwhisper-2.0-turbo",
                engine="crisperwhisper",
                device="cpu",
            ),
            _paths(tmp_path),
            build_engine_registry(),
            _CPU,
            license_policy=policy,
        )

        assert isinstance(engine, CrisperWhisperAdapter)

    def test_should_leave_an_ungated_model_buildable(
        self, tmp_path: Path,
    ) -> None:
        engine = create_engine_for_active_model(
            _config(engine="nano_parakeet", device="cpu"),
            _paths(tmp_path),
            build_engine_registry(),
            _CPU,
            license_policy=LicensePolicy(FakeConfig()),
        )

        assert isinstance(engine, NanoParakeetAdapter)


class TestModelManagers:
    def test_default_manager_should_list_all_models(self, tmp_path: Path) -> None:
        manager = create_model_manager(
            tmp_path, MagicMock(), AllowAllLicensePolicy(),
        )

        assert {m.name for m in manager.list_models()} == set(ALL_MODELS)

    def test_composite_manager_should_list_all_models(
        self, tmp_path: Path,
    ) -> None:
        manager = build_composite_model_manager(
            tmp_path, MagicMock(),
            active_engine_provider=lambda: "nano_parakeet",
            license_policy=AllowAllLicensePolicy(),
        )

        assert {m.name for m in manager.list_models()} == set(ALL_MODELS)
