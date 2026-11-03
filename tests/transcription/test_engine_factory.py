import importlib.util
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.fakes.config import FakeConfig
from tests.fakes.hardware import FakeHardwareDetector
from tests.fakes.licensing import AllowAllLicensePolicy
from skryb.core.exceptions import ConfigurationError
from skryb.transcription.engine_factory import (
    build_composite_model_manager,
    build_engine_registry,
    create_engine_for_active_model,
    create_model_manager,
    engines_with_model_on_disk,
    resolve_device,
    resolve_engine_id,
    selectable_engine_ids,
    startup_requires_torch,
)
from skryb.transcription.engine_registry import EngineRegistry
from skryb.transcription.engine_types import HardwareProfile
from skryb.transcription.model_info import (
    ALL_MODELS,
    DEFAULT_MODEL,
    PARAKEET_MODELS,
    ModelInfo,
    get_model_info,
)
from skryb.transcription.nano_parakeet_adapter import NanoParakeetAdapter


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


_GPU = FakeHardwareDetector(
    HardwareProfile(
        kind="amd_rocm", device_name="AMD Radeon", gfx_arch="gfx1200",
    )
)
_CPU = FakeHardwareDetector(
    HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)
)

_DUAL_ENGINE_MODEL = ModelInfo(
    name="dual", size_mb=1, description="test", repo_id="example/dual",
    compatible_engines=("nano_parakeet", "nemotron_streaming"),
    engine_priorities={"nano_parakeet": 100, "nemotron_streaming": 50},
)

# Saved only in NeMo's layout, the way NVIDIA ships Parakeet: nano_parakeet
# is its one engine, and it tracks main.
_NEMO_ONLY_MODEL = ModelInfo(
    name="nemo-only", size_mb=1, description="test", repo_id="example/nemo-only",
    compatible_engines=("nano_parakeet",),
)

_PINNED_SHA = "1e28271b79edc97635783bea65abc89195a09ed3"
_PINNED_MODEL = ModelInfo(
    name="pinned", size_mb=1, description="test", repo_id="example/pinned",
    revision=_PINNED_SHA, compatible_engines=("nano_parakeet",),
)


class TestTheNpuEngine:
    """parakeet-ultra puts the encoder on the NPU on a GPU-less laptop."""

    @staticmethod
    def _ranked(kind: str, openvino_installed: bool, model: str | ModelInfo = "parakeet-ultra") -> list[str]:
        from skryb.transcription.engine_selector import rank_engines

        info = model if isinstance(model, ModelInfo) else get_model_info(model)
        descriptors = [
            replace(d, available=openvino_installed) if d.engine_id == "openvino_parakeet" else d
            for d in build_engine_registry().all_descriptors()
        ]
        hardware = HardwareProfile(kind=kind, device_name="test", gfx_arch=None)
        return [c.engine_id for c in rank_engines(info, hardware, descriptors)]

    def test_a_computer_without_a_gpu_runs_it_on_the_npu_engine_first(self) -> None:
        assert self._ranked("cpu", openvino_installed=True) == [
            "openvino_parakeet", "nano_parakeet",
        ]

    def test_without_openvino_installed_it_stays_on_pytorch(self) -> None:
        assert self._ranked("cpu", openvino_installed=False) == ["nano_parakeet"]

    @pytest.mark.parametrize("kind", ["amd_rocm", "nvidia_cuda"])
    def test_a_computer_with_a_gpu_keeps_it_on_pytorch(self, kind: str) -> None:
        assert self._ranked(kind, openvino_installed=True) == ["nano_parakeet"]

    def test_a_model_without_transformers_weights_never_goes_to_the_npu(self) -> None:
        assert self._ranked("cpu", True, model=_NEMO_ONLY_MODEL) == ["nano_parakeet"]

    def test_the_npu_engine_shares_the_parakeet_downloads(self, tmp_path: Path) -> None:
        from skryb.transcription.engine_registry import EngineSpec
        from skryb.transcription.openvino_parakeet_adapter import OpenVinoParakeetAdapter

        engine = build_engine_registry().build("openvino_parakeet", EngineSpec(
            model_name="parakeet-ultra", repo_id="moondream/parakeet-ultra",
            device="cpu", language="en", beam_size=1, model_dir=tmp_path,
        ))

        assert isinstance(engine, OpenVinoParakeetAdapter)
        assert engine.is_model_loaded() is False


_TRANSFORMERS_LAYOUT_PARAKEETS = ["parakeet-ultra"]


class TestTheNativeParakeetEngine:
    """Parakeet Ultra and Bark run on the hand-written gfx1200 engine when its DLL is there."""

    @staticmethod
    def _ranked(gfx_arch: str | None, dll_installed: bool, model: str) -> list[str]:
        from skryb.transcription.engine_selector import rank_engines

        descriptors = [
            replace(d, available=dll_installed) if d.engine_id == "native_parakeet" else d
            for d in build_engine_registry().all_descriptors()
        ]
        hardware = HardwareProfile(kind="amd_rocm", device_name="AMD Radeon", gfx_arch=gfx_arch)
        return [c.engine_id for c in rank_engines(get_model_info(model), hardware, descriptors)]

    @pytest.mark.parametrize("model", _TRANSFORMERS_LAYOUT_PARAKEETS)
    def test_an_rx_9060_xt_runs_it_on_the_native_engine_first(self, model: str) -> None:
        assert self._ranked("gfx1200", dll_installed=True, model=model) == ["native_parakeet", "nano_parakeet"]

    @pytest.mark.parametrize("gfx_arch", ["gfx1100", "gfx1201", None])
    def test_another_amd_gpu_stays_on_pytorch(self, gfx_arch: str | None) -> None:
        assert self._ranked(gfx_arch, dll_installed=True, model="parakeet-ultra") == ["nano_parakeet"]

    def test_without_the_dll_it_stays_on_pytorch(self) -> None:
        assert self._ranked("gfx1200", dll_installed=False, model="parakeet-ultra") == ["nano_parakeet"]

    def test_an_rx_9060_xt_runs_phonon_on_it_ahead_of_fermions_cpu_engine(self) -> None:
        assert self._ranked("gfx1200", dll_installed=True, model="phonon-2.5-english")[:2] == [
            "native_parakeet", "phonon2_cpu",
        ]

    @pytest.mark.parametrize(("gfx_arch", "dll_installed"), [("gfx1100", True), ("gfx1200", False)])
    def test_phonon_stays_on_fermions_cpu_engine_anywhere_else(self, gfx_arch: str, dll_installed: bool) -> None:
        assert self._ranked(gfx_arch, dll_installed=dll_installed, model="phonon-2.5-english")[0] == "phonon2_cpu"

    def test_it_shares_the_parakeet_downloads(self, tmp_path: Path) -> None:
        from skryb.transcription.engine_registry import EngineSpec
        from skryb.transcription.native_parakeet_adapter import NativeParakeetAdapter

        engine = build_engine_registry().build("native_parakeet", EngineSpec(
            model_name="parakeet-ultra", repo_id="moondream/parakeet-ultra",
            device="cuda", language="en", beam_size=1, model_dir=tmp_path,
        ))
        manager = build_composite_model_manager(
            tmp_path, MagicMock(), lambda: "native_parakeet", AllowAllLicensePolicy(),
        )

        assert isinstance(engine, NativeParakeetAdapter)
        assert engine.is_model_loaded() is False
        assert "parakeet-ultra" in {m.name for m in manager.list_models()}

    def test_the_cpu_fallback_after_it_fails_runs_parakeet_on_pytorch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("skryb.transcription.engine_factory.native_parakeet_dll", lambda: tmp_path / "npk.dll")
        find_spec = importlib.util.find_spec
        monkeypatch.setattr(  # without the optional NPU engine, PyTorch is the CPU's only Parakeet engine
            "importlib.util.find_spec", lambda name, *rest: None if name == "openvino" else find_spec(name, *rest),
        )

        engine = create_engine_for_active_model(
            _config(model_size="parakeet-ultra", device="cuda"), _paths(tmp_path), build_engine_registry(),
            _GPU, license_policy=AllowAllLicensePolicy(), device_override="cpu",
        )

        assert isinstance(engine, NanoParakeetAdapter)
        assert engine._device == "cpu"

    def test_startup_skips_torch_when_an_earlier_launch_found_an_rx_9060_xt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("skryb.transcription.engine_factory.native_parakeet_dll", lambda: tmp_path / "npk.dll")
        config = FakeConfig({"transcription.model_size": "parakeet-ultra", "transcription.engine": "auto"})
        remembered = HardwareProfile(kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200")

        assert selectable_engine_ids(config, build_engine_registry(), tmp_path, remembered) == ("native_parakeet",)
        assert startup_requires_torch(config, build_engine_registry(), tmp_path, remembered) is False

    def test_startup_still_warms_torch_when_no_gpu_probe_is_remembered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("skryb.transcription.engine_factory.native_parakeet_dll", lambda: tmp_path / "npk.dll")
        config = FakeConfig({"transcription.model_size": "parakeet-ultra", "transcription.engine": "auto"})

        assert startup_requires_torch(config, build_engine_registry(), tmp_path) is True


class TestTheNativeEnginesOnNvidia:
    """The native engines run their NVIDIA build on an RTX 30, 40 or 50 series card."""

    _RTX_5060 = HardwareProfile(kind="nvidia_cuda", device_name="NVIDIA GeForce RTX 5060", gfx_arch="sm_120")

    @staticmethod
    def _ranked(gfx_arch: str | None, model: str) -> list[str]:
        from skryb.transcription.engine_selector import rank_engines

        descriptors = [replace(d, available=True) for d in build_engine_registry().all_descriptors()]
        hardware = HardwareProfile(kind="nvidia_cuda", device_name="NVIDIA GeForce RTX", gfx_arch=gfx_arch)
        return [c.engine_id for c in rank_engines(get_model_info(model), hardware, descriptors)]

    @pytest.mark.parametrize("gfx_arch", ["sm_120", "sm_89", "sm_86"])
    @pytest.mark.parametrize(("model", "native"), [
        ("parakeet-ultra", "native_parakeet"),
        ("qwen3-asr-1.7b", "native_qwen3_asr"),
    ])
    def test_an_rtx_30_40_or_50_card_runs_each_model_on_the_native_engine_first(
        self, gfx_arch: str, model: str, native: str,
    ) -> None:
        assert self._ranked(gfx_arch, model)[0] == native

    @pytest.mark.parametrize("gfx_arch", ["sm_75", "sm_80", None])
    def test_an_older_or_unprobed_nvidia_card_stays_on_pytorch(self, gfx_arch: str | None) -> None:
        assert self._ranked(gfx_arch, "parakeet-ultra") == ["nano_parakeet"]

    def test_the_nvidia_build_alone_makes_the_engine_available(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / "npk_cuda.dll").write_bytes(b"MZ")
        monkeypatch.setenv("NPK_CUDA_DLL", str(tmp_path / "npk_cuda.dll"))

        assert build_engine_registry().descriptor("native_parakeet").available

    @pytest.mark.parametrize(("hardware", "dll"), [
        (_RTX_5060, "npk_cuda.dll"),
        (HardwareProfile(kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200"), "npk.dll"),
    ])
    def test_the_adapter_loads_the_build_for_the_gpu_it_was_picked_for(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hardware: HardwareProfile, dll: str,
    ) -> None:
        from skryb.transcription.native_parakeet_adapter import NativeParakeetAdapter

        for name, variable in (("npk.dll", "NPK_DLL"), ("npk_cuda.dll", "NPK_CUDA_DLL")):
            (tmp_path / name).write_bytes(b"MZ")
            monkeypatch.setenv(variable, str(tmp_path / name))

        engine = create_engine_for_active_model(
            _config(model_size="parakeet-ultra"), _paths(tmp_path), build_engine_registry(),
            FakeHardwareDetector(hardware), license_policy=AllowAllLicensePolicy(),
        )

        assert isinstance(engine, NativeParakeetAdapter)
        assert engine._dll_path == tmp_path / dll


class TestBuildEngineRegistry:
    def test_should_register_only_surviving_engines(self) -> None:
        registry = build_engine_registry()

        assert {d.engine_id for d in registry.all_descriptors()} == {
            "nano_parakeet", "nemotron_streaming", "phonon2_cpu", "native_qwen3_asr", "openvino_parakeet",
            "native_parakeet", "qwen3_asr", "openvino_qwen3_asr",
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
        _fake_snapshot(tmp_path, _NEMO_ONLY_MODEL.repo_id)

        result = engines_with_model_on_disk(_NEMO_ONLY_MODEL, tmp_path)

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
        assert engine._repo_id == "moondream/parakeet-ultra"

    def test_device_override_should_force_cpu(self, tmp_path: Path) -> None:
        engine = create_engine_for_active_model(
            _config(device="cuda"), _paths(tmp_path), build_engine_registry(),
            _GPU, license_policy=AllowAllLicensePolicy(), device_override="cpu",
        )

        assert isinstance(engine, NanoParakeetAdapter)
        assert engine._device == "cpu"

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


def _torch_free(registry: EngineRegistry) -> EngineRegistry:
    return EngineRegistry(
        descriptors=[
            replace(descriptor, requires_torch=False)
            for descriptor in registry.all_descriptors()
        ],
        builders={},
    )


class TestSelectableEngineIds:
    """Startup reads this before importing anything, so it has to answer
    from the config and the catalogue alone.
    """

    def test_should_name_only_the_explicitly_configured_engine(self) -> None:
        ids = selectable_engine_ids(
            _config(engine="qwen3_asr"), build_engine_registry(),
        )

        assert ids == ("qwen3_asr",)

    def test_should_offer_the_models_compatible_engines_in_auto_mode(
        self,
    ) -> None:
        ids = selectable_engine_ids(_config(), build_engine_registry())

        assert ids == PARAKEET_MODELS[DEFAULT_MODEL].compatible_engines

    def test_should_widen_to_every_engine_for_an_unknown_model(self) -> None:
        registry = build_engine_registry()

        ids = selectable_engine_ids(
            _config(model_size="retired-model"), registry,
        )

        assert set(ids) == {d.engine_id for d in registry.all_descriptors()}


class TestStartupRequiresTorch:
    def test_should_require_torch_for_the_shipped_engines(self) -> None:
        assert startup_requires_torch(_config(), build_engine_registry()) is True

    def test_should_not_require_torch_when_every_candidate_declares_none(
        self,
    ) -> None:
        registry = _torch_free(build_engine_registry())

        assert startup_requires_torch(_config(), registry) is False


class TestResolveDevice:
    def test_should_return_the_explicitly_configured_device(self) -> None:
        device = resolve_device(
            _config(device="cpu"), build_engine_registry(), _GPU,
            ("nano_parakeet",),
        )

        assert device == "cpu"

    def test_should_resolve_auto_from_the_probe_for_a_torch_engine(
        self,
    ) -> None:
        device = resolve_device(
            _config(), build_engine_registry(), _GPU, ("nano_parakeet",),
        )

        assert device == "cuda"

    def test_should_leave_auto_to_a_torch_free_engine_without_probing(
        self,
    ) -> None:
        detector = FakeHardwareDetector()

        device = resolve_device(
            _config(), _torch_free(build_engine_registry()), detector,
            ("nano_parakeet",),
        )

        assert device == "auto"
        assert detector.detect_calls == 0


class TestEngineResolutionSkipsTheProbe:
    """The probe imports torch. When every engine that could drive the
    model runs on every hardware kind, ranking cannot change with the
    answer - so it must not be asked for one.
    """

    def test_should_resolve_a_hardware_agnostic_model_without_probing(
        self,
    ) -> None:
        detector = FakeHardwareDetector()

        engine_id = resolve_engine_id(
            _config(), _NEMO_ONLY_MODEL, detector, build_engine_registry(),
        )

        assert engine_id == "nano_parakeet"
        assert detector.detect_calls == 0

    def test_should_probe_when_a_candidate_is_gpu_only(self) -> None:
        detector = FakeHardwareDetector()

        resolve_engine_id(
            _config(), _DUAL_ENGINE_MODEL, detector, build_engine_registry(),
        )

        assert detector.detect_calls == 1


class TestPhonon2OnTheCpuEngine:
    """Phonon-2.5 English runs on Fermion's CPU engine whatever GPU the machine has."""

    def _installed(self, models_dir: Path) -> Path:
        model_info = get_model_info("phonon-2.5-english")
        repo = models_dir / "models--FermionResearch--Phonon-2.5-English"
        snapshot = repo / "snapshots" / model_info.revision
        snapshot.mkdir(parents=True)
        (snapshot / "phonon-2.5-en.bps.tar.zst").write_bytes(b"zstd tar")
        (snapshot / "config.json").write_text("{}")
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(model_info.revision)
        return snapshot

    @pytest.mark.parametrize("hardware", [_CPU, _GPU])
    def test_auto_keeps_it_on_fermions_engine_without_a_tuned_npu(self, hardware) -> None:
        engine_id = resolve_engine_id(
            _config(model_size="phonon-2.5-english"), get_model_info("phonon-2.5-english"),
            hardware, build_engine_registry(),
        )

        assert engine_id == "phonon2_cpu"

    def _ranked_with_openvino(self, hardware: HardwareProfile) -> tuple:
        from skryb.transcription.engine_selector import rank_engines

        descriptors = [
            replace(d, available=True) if d.engine_id == "openvino_parakeet" else d
            for d in build_engine_registry().all_descriptors()
        ]
        return rank_engines(get_model_info("phonon-2.5-english"), hardware, descriptors)

    def test_a_computer_without_a_gpu_or_a_tuned_npu_can_still_choose_the_npu_engine(self) -> None:
        hardware = HardwareProfile(kind="cpu", device_name="Intel Core Ultra 7 268V", gfx_arch=None)

        assert [c.engine_id for c in self._ranked_with_openvino(hardware)] == ["phonon2_cpu", "openvino_parakeet"]

    def test_a_core_series_3_npu_puts_the_npu_engine_first(self) -> None:
        hardware = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None, npu="FD3E")

        ranked = self._ranked_with_openvino(hardware)

        assert [c.engine_id for c in ranked] == ["openvino_parakeet", "phonon2_cpu"]
        assert ranked[0].reason == "Recommended for your Intel NPU"

    def test_a_tuned_npu_without_openvino_installed_stays_on_fermions_engine(self) -> None:
        from skryb.transcription.engine_selector import rank_engines

        descriptors = [
            replace(d, available=False) if d.engine_id == "openvino_parakeet" else d
            for d in build_engine_registry().all_descriptors()
        ]
        hardware = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None, npu="FD3E")

        ranked = rank_engines(get_model_info("phonon-2.5-english"), hardware, descriptors)

        assert [c.engine_id for c in ranked] == ["phonon2_cpu"]
        assert ranked[0].reason == "Recommended for your CPU"

    @pytest.mark.parametrize("hardware", [_CPU, _GPU])
    def test_the_engine_opens_the_pinned_snapshot_on_any_hardware(
        self, tmp_path: Path, hardware: FakeHardwareDetector,
    ) -> None:
        from skryb.transcription.phonon2_cpu_adapter import Phonon2CpuAdapter

        snapshot = self._installed(tmp_path)

        engine = create_engine_for_active_model(
            _config(model_size="phonon-2.5-english"), _paths(tmp_path),
            build_engine_registry(), hardware,
            license_policy=AllowAllLicensePolicy(),
        )

        assert isinstance(engine, Phonon2CpuAdapter)
        assert engine._snapshot == snapshot

    def test_a_snapshot_at_another_commit_is_not_opened(self, tmp_path: Path) -> None:
        snapshot = self._installed(tmp_path)
        (snapshot.parent.parent / "refs" / "main").write_text("0" * 40)

        assert engines_with_model_on_disk(get_model_info("phonon-2.5-english"), tmp_path) == frozenset()

    def test_the_composite_manager_downloads_it_through_its_own_child(
        self, tmp_path: Path,
    ) -> None:
        manager = build_composite_model_manager(
            tmp_path, MagicMock(), lambda: "nano_parakeet", AllowAllLicensePolicy(),
        )
        self._installed(tmp_path)

        assert manager.has_model("phonon-2.5-english") is True
        assert manager.is_installed("phonon-2.5-english") is True


class TestQwen3StartupStaysTorchFreeOnTheRx9060Xt:
    """Qwen3-ASR has a torch engine for everyone and a torch-free one for gfx1200.
    A launch decides about torch before probing, so it reads the probe a past launch saved."""

    _RX_9060_XT = HardwareProfile(kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200")

    @pytest.fixture(autouse=True)
    def _native_dll_present(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "skryb.transcription.engine_factory.native_engine_dll", lambda: tmp_path / "qasr.dll",
        )

    def _config(self) -> FakeConfig:
        return FakeConfig({"transcription.model_size": "qwen3-asr-1.7b", "transcription.engine": "auto"})

    def test_with_the_remembered_rx_9060_xt_it_picks_the_native_engine_and_skips_torch(self, tmp_path: Path) -> None:
        registry = build_engine_registry()

        assert selectable_engine_ids(self._config(), registry, tmp_path, self._RX_9060_XT) == ("native_qwen3_asr",)
        assert startup_requires_torch(self._config(), registry, tmp_path, self._RX_9060_XT) is False

    @pytest.mark.parametrize("hardware", [
        HardwareProfile(kind="nvidia_cuda", device_name="NVIDIA GeForce RTX 4070", gfx_arch=None),
        HardwareProfile(kind="amd_rocm", device_name="AMD Radeon RX 7900 XTX", gfx_arch="gfx1100"),
    ])
    def test_on_any_other_gpu_it_runs_on_transformers_and_warms_torch(self, tmp_path: Path, hardware: HardwareProfile) -> None:
        registry = build_engine_registry()

        assert selectable_engine_ids(self._config(), registry, tmp_path, hardware) == ("qwen3_asr",)
        assert startup_requires_torch(self._config(), registry, tmp_path, hardware) is True

    @pytest.mark.parametrize("npu", [None, "FD3E"])
    def test_on_a_computer_without_a_gpu_it_runs_on_openvino_and_skips_torch(self, tmp_path: Path, npu: str | None) -> None:
        registry = build_engine_registry()
        hardware = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None, npu=npu)

        assert selectable_engine_ids(self._config(), registry, tmp_path, hardware) == ("openvino_qwen3_asr",)
        assert startup_requires_torch(self._config(), registry, tmp_path, hardware) is False

    def test_before_any_probe_has_been_saved_it_plans_for_torch(self, tmp_path: Path) -> None:
        assert startup_requires_torch(self._config(), build_engine_registry(), tmp_path) is True
