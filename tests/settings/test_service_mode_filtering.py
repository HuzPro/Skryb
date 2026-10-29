import pytest

from tests.settings.test_service import _build_service
from skryb.transcription import engine_factory
from skryb.output.output_mode import OutputMode
from skryb.settings.settings_draft import ChangeSet, SettingsDraft
from skryb.settings.transcription_mode import TranscriptionMode
from skryb.transcription.engine_factory import build_engine_registry
from skryb.transcription.engine_types import HardwareProfile
from skryb.transcription.model_info import ALL_MODELS

_PARAKEET = "parakeet-ultra"
_NEMOTRON = "nemotron-speech-streaming-en-0.6b"
_CRISPERWHISPER = {
    "crisperwhisper-2.0-turbo",
    "crisperwhisper-2.0-large",
}

def _gpu(kind: str) -> HardwareProfile:
    return HardwareProfile(kind=kind, device_name=kind, gfx_arch=None)


def _draft(mode: TranscriptionMode) -> SettingsDraft:
    return SettingsDraft(
        model_size=_PARAKEET, language="en", device="auto", engine="auto",
        output_mode=OutputMode.DIRECT_PASTE, transcription_mode=mode,
    )


class TestSettingsDraftMode:
    def test_should_default_transcription_mode_to_batch(self):
        draft = SettingsDraft(
            model_size=_PARAKEET, language="en", device="auto", engine="auto",
            output_mode=OutputMode.DIRECT_PASTE,
        )

        assert draft.transcription_mode is TranscriptionMode.BATCH

    def test_should_flag_mode_change_in_changeset(self):
        changes = ChangeSet.between(
            _draft(TranscriptionMode.BATCH), _draft(TranscriptionMode.STREAMING),
        )

        assert changes.mode_changed is True
        assert changes.any_changed is True


class TestServiceModeRoundTrip:
    def test_should_read_streaming_mode_from_config(self):
        service, *_ = _build_service(
            config_overrides={"transcription.mode": "streaming"},
        )

        assert service.current_draft().transcription_mode is (
            TranscriptionMode.STREAMING
        )

    def test_should_persist_mode_on_apply(self):
        service, config, *_ = _build_service()

        service.apply_draft(_draft(TranscriptionMode.STREAMING))

        assert config.get("transcription.mode") == "streaming"


class TestServiceModeFiltering:
    def test_should_list_all_models_in_batch_mode(self):
        service, *_ = _build_service(
            models=ALL_MODELS, registry=build_engine_registry(),
        )

        names = {view.name for view in service.list_models()}

        assert names == {
            _PARAKEET, "phonon-2.5-english", *_CRISPERWHISPER, "qwen3-asr-1.7b",
        }

    def test_should_list_only_streaming_models_in_streaming_mode(self):
        service, *_ = _build_service(
            models=ALL_MODELS, registry=build_engine_registry(),
            config_overrides={"transcription.mode": "streaming"},
        )

        names = {view.name for view in service.list_models()}

        assert names == {
            _PARAKEET,
            _NEMOTRON,
        }

    def test_should_report_streaming_capability_per_model(self):
        service, *_ = _build_service(registry=build_engine_registry())

        assert service.model_supports_streaming(_PARAKEET) is True
        assert service.model_supports_streaming(_NEMOTRON) is True

    def test_should_report_batch_capability_per_model(self):
        service, *_ = _build_service(registry=build_engine_registry())

        assert service.model_supports_batch(_PARAKEET) is True
        assert (
            service.model_supports_batch(_NEMOTRON)
            is False
        )

    def test_first_run_catalog_routing_should_never_be_empty(self):
        profiles = [
            _gpu("cpu"),
            _gpu("nvidia_cuda"),
            _gpu("amd_rocm"),
        ]
        for profile in profiles:
            for mode in ("batch", "streaming"):
                service, *_ = _build_service(
                    models=ALL_MODELS,
                    registry=build_engine_registry(),
                    hardware=profile,
                    config_overrides={"transcription.mode": mode},
                )

                assert service.list_models(), (profile.kind, mode)

    def test_gpu_batch_should_hide_streaming_only_nemotron(self):
        service, *_ = _build_service(
            models=ALL_MODELS,
            registry=build_engine_registry(),
            hardware=_gpu("nvidia_cuda"),
        )

        names = {view.name for view in service.list_models()}

        assert names == {_PARAKEET, "phonon-2.5-english", *_CRISPERWHISPER, "qwen3-asr-1.7b"}

    def test_gpu_streaming_should_list_parakeet_and_nemotron(self):
        service, *_ = _build_service(
            models=ALL_MODELS,
            registry=build_engine_registry(),
            hardware=_gpu("amd_rocm"),
            config_overrides={"transcription.mode": "streaming"},
        )

        names = {view.name for view in service.list_models()}

        assert names == {
            _PARAKEET,
            _NEMOTRON,
        }


class TestStreamingHardwareGate:
    """Streaming is GPU-only, so the service reports whether the box can run it."""

    def test_should_support_streaming_on_an_nvidia_gpu(self):
        service, *_ = _build_service(hardware=_gpu("nvidia_cuda"))

        assert service.supports_streaming_hardware() is True

    def test_should_support_streaming_on_an_amd_gpu(self):
        service, *_ = _build_service(hardware=_gpu("amd_rocm"))

        assert service.supports_streaming_hardware() is True

    def test_should_not_support_streaming_on_a_cpu_only_box(self):
        service, *_ = _build_service(hardware=_gpu("cpu"))

        assert service.supports_streaming_hardware() is False

    def test_should_not_support_streaming_when_hardware_is_unknown(self):
        service, *_ = _build_service()  # hardware=None

        assert service.supports_streaming_hardware() is False


class TestQwen3RunsOnEveryComputer:
    """The native engine's kernels are compiled for gfx1200 and live in qasr.dll;
    everywhere else Qwen3-ASR runs on its transformers engine."""

    @staticmethod
    def _service(monkeypatch, tmp_path, gfx_arch, *, dll_installed=True, kind="amd_rocm"):
        dll = tmp_path / "qasr.dll"
        monkeypatch.setattr(
            engine_factory, "native_engine_dll", lambda: dll if dll_installed else None,
        )
        hardware = None if kind is None else HardwareProfile(kind=kind, device_name="GPU", gfx_arch=gfx_arch)
        service, *_ = _build_service(
            models=ALL_MODELS, registry=build_engine_registry(), hardware=hardware,
        )
        return service

    def _engines(self, monkeypatch, tmp_path, gfx_arch, **kwargs):
        service = self._service(monkeypatch, tmp_path, gfx_arch, **kwargs)
        assert "qwen3-asr-1.7b" in {view.name for view in service.list_models()}
        return [candidate.engine_id for candidate in service.engine_candidates_for("qwen3-asr-1.7b")]

    @pytest.mark.parametrize("gfx_arch", ["gfx1200", "GFX1200", "gfx1200:sramecc-:xnack-", " gfx1200 "])
    def test_the_rx_9060_xt_runs_it_on_the_native_engine_first(self, monkeypatch, tmp_path, gfx_arch):
        assert self._engines(monkeypatch, tmp_path, gfx_arch) == ["native_qwen3_asr", "qwen3_asr"]

    @pytest.mark.parametrize("gfx_arch", ["gfx1201", "gfx1100", "gfx1101", "gfx90a", "", None, "gfx12000"])
    def test_other_amd_gpus_run_it_on_transformers(self, monkeypatch, tmp_path, gfx_arch):
        assert self._engines(monkeypatch, tmp_path, gfx_arch) == ["qwen3_asr"]

    def test_without_the_native_dll_it_runs_on_transformers(self, monkeypatch, tmp_path):
        assert self._engines(monkeypatch, tmp_path, "gfx1200", dll_installed=False) == ["qwen3_asr"]

    @pytest.mark.parametrize("kind", ["cpu", "nvidia_cuda"])
    def test_nvidia_and_cpu_run_it_on_transformers(self, monkeypatch, tmp_path, kind):
        assert self._engines(monkeypatch, tmp_path, None, kind=kind) == ["qwen3_asr"]

    def test_every_model_is_listed_before_the_hardware_is_known(self, monkeypatch, tmp_path):
        service = self._service(monkeypatch, tmp_path, None, dll_installed=False, kind=None)

        assert "qwen3-asr-1.7b" in {view.name for view in service.list_models()}
