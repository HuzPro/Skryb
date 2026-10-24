import pytest

from skryb.transcription.engine_types import (
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
            engine_id="nano_parakeet",
            display_name="nano-parakeet (PyTorch)",
            supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
            base_priority=50,
        )

        assert descriptor.engine_id == "nano_parakeet"
        assert descriptor.display_name == "nano-parakeet (PyTorch)"
        assert descriptor.supported_hardware == ("nvidia_cuda", "amd_rocm", "cpu")
        assert descriptor.base_priority == 50


class TestEngineCandidate:
    def test_should_create_with_required_fields(self):
        candidate = EngineCandidate(
            engine_id="nano_parakeet",
            display_name="nano-parakeet (PyTorch)",
            score=100,
            is_recommended=True,
            reason="Recommended for your AMD ROCm GPU (gfx1200)",
        )

        assert candidate.engine_id == "nano_parakeet"
        assert candidate.display_name == "nano-parakeet (PyTorch)"
        assert candidate.score == 100
        assert candidate.is_recommended is True
        assert candidate.reason.startswith("Recommended")
