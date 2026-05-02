import pytest

from voicetotext.transcription.engine_types import (
    EngineCandidate,
    EngineDescriptor,
    HardwareProfile,
)


class TestHardwareProfile:
    def test_should_create_with_required_fields(self):
        profile = HardwareProfile(
            kind="cpu", device_name="CPU", gfx_arch=None,
        )

        assert profile.kind == "cpu"
        assert profile.device_name == "CPU"
        assert profile.gfx_arch is None

    def test_should_be_frozen(self):
        profile = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)

        with pytest.raises(AttributeError):
            profile.kind = "amd_rocm"  # type: ignore[misc]


class TestEngineDescriptor:
    def test_should_create_with_required_fields(self):
        descriptor = EngineDescriptor(
            engine_id="transformers",
            display_name="Transformers (HuggingFace)",
            supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
            base_priority=50,
        )

        assert descriptor.engine_id == "transformers"
        assert descriptor.display_name == "Transformers (HuggingFace)"
        assert descriptor.supported_hardware == ("nvidia_cuda", "amd_rocm", "cpu")
        assert descriptor.base_priority == 50


class TestEngineCandidate:
    def test_should_create_with_required_fields(self):
        candidate = EngineCandidate(
            engine_id="faster_whisper",
            display_name="faster-whisper (CTranslate2)",
            score=100,
            is_recommended=True,
            reason="Recommended for your AMD ROCm GPU (gfx1200)",
        )

        assert candidate.engine_id == "faster_whisper"
        assert candidate.display_name == "faster-whisper (CTranslate2)"
        assert candidate.score == 100
        assert candidate.is_recommended is True
        assert candidate.reason.startswith("Recommended")
