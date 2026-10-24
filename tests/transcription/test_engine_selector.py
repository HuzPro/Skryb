from skryb.transcription.engine_selector import rank_engines
from skryb.transcription.engine_types import (
    EngineDescriptor,
    HardwareProfile,
)
from skryb.transcription.model_info import ModelInfo


_PARAKEET = EngineDescriptor(
    engine_id="nano_parakeet",
    display_name="nano-parakeet",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=90,
)
_NEMOTRON = EngineDescriptor(
    engine_id="nemotron_streaming",
    display_name="Nemotron Streaming",
    supported_hardware=("nvidia_cuda", "amd_rocm"),
    base_priority=85,
)
_GPU = HardwareProfile(
    kind="amd_rocm", device_name="AMD Radeon", gfx_arch="gfx1200",
)
_CPU = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)


def _dual_engine_model() -> ModelInfo:
    return ModelInfo(
        name="dual", size_mb=1, description="test", repo_id="example/dual",
        compatible_engines=("nano_parakeet", "nemotron_streaming"),
        engine_priorities={"nano_parakeet": 100, "nemotron_streaming": 50},
    )


class TestRankEngines:
    def test_should_order_by_model_priority(self) -> None:
        candidates = rank_engines(
            _dual_engine_model(), _GPU, [_NEMOTRON, _PARAKEET],
        )

        assert [candidate.engine_id for candidate in candidates] == [
            "nano_parakeet", "nemotron_streaming",
        ]
        assert candidates[0].is_recommended is True

    def test_should_drop_engine_not_supported_by_hardware(self) -> None:
        candidates = rank_engines(
            _dual_engine_model(), _CPU, [_NEMOTRON, _PARAKEET],
        )

        assert [candidate.engine_id for candidate in candidates] == [
            "nano_parakeet",
        ]

    def test_should_drop_engine_not_declared_by_model(self) -> None:
        model = ModelInfo(
            name="batch", size_mb=1, description="test",
            compatible_engines=("nano_parakeet",),
        )

        candidates = rank_engines(model, _GPU, [_NEMOTRON, _PARAKEET])

        assert [candidate.engine_id for candidate in candidates] == [
            "nano_parakeet",
        ]

    def test_should_return_empty_when_no_engine_is_compatible(self) -> None:
        model = ModelInfo(name="none", size_mb=1, description="test")

        assert rank_engines(model, _GPU, [_NEMOTRON, _PARAKEET]) == ()
