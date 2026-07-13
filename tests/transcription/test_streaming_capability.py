from voicetotext.transcription.engine_factory import build_engine_registry
from voicetotext.transcription.engine_selector import (
    model_supports_batch,
    model_supports_streaming,
    rank_engines,
)
from voicetotext.transcription.engine_types import (
    EngineDescriptor,
    HardwareProfile,
)
from voicetotext.transcription.model_info import ModelInfo, get_model_info


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


class TestNemotronStreamingDescriptor:
    def test_should_register_nemotron_streaming_as_streaming_only(self):
        by_id = {d.engine_id: d for d in _descriptors()}

        assert by_id["nemotron_streaming"].supports_streaming is True
        assert by_id["nemotron_streaming"].streaming_only is True

    def test_should_be_gpu_only(self):
        by_id = {d.engine_id: d for d in _descriptors()}

        assert "cpu" not in by_id["nemotron_streaming"].supported_hardware


class TestModelSupportsBatch:
    def test_should_report_nemotron_not_batch_capable(self):
        assert model_supports_batch(
            get_model_info("nemotron-speech-streaming-en-0.6b"), _descriptors(),
        ) is False

    def test_should_report_nemotron_streaming_capable(self):
        assert model_supports_streaming(
            get_model_info("nemotron-speech-streaming-en-0.6b"), _descriptors(),
        ) is True

    def test_should_report_parakeet_batch_capable(self):
        assert model_supports_batch(
            get_model_info("parakeet-tdt-0.6b-v3"), _descriptors(),
        ) is True

    def test_should_report_whisper_batch_capable(self):
        assert model_supports_batch(
            get_model_info("base"), _descriptors(),
        ) is True


class TestSelectorUnaffectedByStreamingFlag:
    def test_streaming_flag_should_not_change_ranking_order(self):
        # rank_engines must order purely by priority and hardware, never by
        # the descriptor's supports_streaming flag. Two descriptor sets that
        # differ ONLY in that flag must rank a dual-engine model identically.
        rocm = HardwareProfile(
            kind="amd_rocm", device_name="AMD", gfx_arch="gfx1200",
        )
        model = ModelInfo(
            name="dual", size_mb=1, description="test", repo_id="example/dual",
            compatible_engines=("transformers", "nano_parakeet"),
            engine_priorities={"nano_parakeet": 100, "transformers": 50},
        )

        def descriptors(streaming: bool) -> list[EngineDescriptor]:
            return [
                EngineDescriptor(
                    engine_id="transformers", display_name="t",
                    supported_hardware=("amd_rocm",), base_priority=50,
                ),
                EngineDescriptor(
                    engine_id="nano_parakeet", display_name="p",
                    supported_hardware=("amd_rocm",), base_priority=90,
                    supports_streaming=streaming,
                ),
            ]

        order_with = [c.engine_id for c in rank_engines(model, rocm, descriptors(True))]
        order_without = [
            c.engine_id for c in rank_engines(model, rocm, descriptors(False))
        ]

        assert order_with == order_without == ["nano_parakeet", "transformers"]
