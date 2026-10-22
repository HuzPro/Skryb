from tests.settings.test_service import _build_service
from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.transcription_mode import TranscriptionMode
from voicetotext.transcription.engine_factory import build_engine_registry
from voicetotext.transcription.engine_types import HardwareProfile
from voicetotext.transcription.model_info import ALL_MODELS

_PARAKEET = "parakeet-tdt-0.6b-v3"
_PARAKEET_ULTRA = "parakeet-ultra"
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
            _PARAKEET, _PARAKEET_ULTRA, "phonon-2", *_CRISPERWHISPER, "qwen3-asr-1.7b",
        }

    def test_should_list_only_streaming_models_in_streaming_mode(self):
        service, *_ = _build_service(
            models=ALL_MODELS, registry=build_engine_registry(),
            config_overrides={"transcription.mode": "streaming"},
        )

        names = {view.name for view in service.list_models()}

        assert names == {
            _PARAKEET,
            _PARAKEET_ULTRA,
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

        assert names == {
            _PARAKEET, _PARAKEET_ULTRA, "phonon-2", *_CRISPERWHISPER, "qwen3-asr-1.7b",
        }

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
            _PARAKEET_ULTRA,
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
