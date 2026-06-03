from voicetotext.transcription.engine_factory import build_engine_registry
from voicetotext.transcription.engine_selector import (
    model_supports_streaming,
    rank_engines,
)
from voicetotext.transcription.engine_types import (
    EngineDescriptor,
    HardwareProfile,
)
from voicetotext.transcription.model_info import get_model_info


def _descriptors() -> list[EngineDescriptor]:
    return list(build_engine_registry().all_descriptors())


class TestEngineDescriptorStreamingFlag:
    def test_should_default_supports_streaming_to_false(self):
        descriptor = EngineDescriptor(
            engine_id="transformers", display_name="x",
            supported_hardware=("cpu",), base_priority=1,
        )

        assert descriptor.supports_streaming is False

    def test_should_store_explicit_supports_streaming(self):
        descriptor = EngineDescriptor(
            engine_id="nano_parakeet", display_name="x",
            supported_hardware=("cpu",), base_priority=1,
            supports_streaming=True,
        )

        assert descriptor.supports_streaming is True


class TestRegistryStreamingDescriptors:
    def test_should_mark_nano_parakeet_as_streaming_capable(self):
        by_id = {d.engine_id: d for d in _descriptors()}

        assert by_id["nano_parakeet"].supports_streaming is True

    def test_should_not_mark_other_engines_as_streaming_capable(self):
        by_id = {d.engine_id: d for d in _descriptors()}

        assert by_id["transformers"].supports_streaming is False
        assert by_id["faster_whisper"].supports_streaming is False
        assert by_id["granite_speech"].supports_streaming is False


class TestModelSupportsStreaming:
    def test_should_report_parakeet_streaming_capable(self):
        assert model_supports_streaming(
            get_model_info("parakeet-tdt-0.6b-v3"), _descriptors(),
        ) is True

    def test_should_report_whisper_not_streaming_capable(self):
        assert model_supports_streaming(
            get_model_info("base"), _descriptors(),
        ) is False

    def test_should_report_granite_not_streaming_capable(self):
        assert model_supports_streaming(
            get_model_info("granite-speech-4.1-2b"), _descriptors(),
        ) is False


class TestSelectorUnaffectedByStreamingFlag:
    def test_should_still_recommend_faster_whisper_for_whisper_on_rocm(self):
        hardware = HardwareProfile(
            kind="amd_rocm", device_name="AMD", gfx_arch="gfx1200",
        )

        candidates = rank_engines(
            get_model_info("base"), hardware, _descriptors(),
        )

        assert candidates[0].engine_id == "faster_whisper"
