from voicetotext.transcription.engine_selector import rank_engines
from voicetotext.transcription.engine_types import (
    EngineDescriptor,
    HardwareProfile,
)
from voicetotext.transcription.model_info import ModelInfo


_TRANSFORMERS = EngineDescriptor(
    engine_id="transformers",
    display_name="Transformers (HuggingFace)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=50,
)

_FASTER_WHISPER = EngineDescriptor(
    engine_id="faster_whisper",
    display_name="faster-whisper (CTranslate2)",
    supported_hardware=("nvidia_cuda", "amd_rocm", "cpu"),
    base_priority=80,
)

_ROCM_PROFILE = HardwareProfile(
    kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200",
)
_CUDA_PROFILE = HardwareProfile(
    kind="nvidia_cuda", device_name="NVIDIA RTX 4090", gfx_arch=None,
)
_CPU_PROFILE = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)


def _whisper_model(
    *,
    compatible_engines: tuple[str, ...] = ("transformers", "faster_whisper"),
    engine_priorities: dict[str, int] | None = None,
) -> ModelInfo:
    return ModelInfo(
        name="large-v3-turbo", size_mb=1620, description="test",
        repo_id="openai/whisper-large-v3-turbo",
        ct2_repo_id="Systran/faster-whisper-large-v3-turbo",
        compatible_engines=compatible_engines,
        engine_priorities=engine_priorities or {},
    )


class TestRankEngines:
    def test_should_recommend_faster_whisper_on_amd_rocm_for_whisper_model(self):
        model = _whisper_model(
            engine_priorities={"faster_whisper": 100, "transformers": 50},
        )

        candidates = rank_engines(
            model, _ROCM_PROFILE, [_TRANSFORMERS, _FASTER_WHISPER],
        )

        assert candidates[0].engine_id == "faster_whisper"
        assert candidates[0].is_recommended is True
        assert "AMD" in candidates[0].reason or "ROCm" in candidates[0].reason

    def test_should_recommend_transformers_when_faster_whisper_not_in_descriptors(self):
        model = _whisper_model()

        candidates = rank_engines(model, _CPU_PROFILE, [_TRANSFORMERS])

        assert len(candidates) == 1
        assert candidates[0].engine_id == "transformers"
        assert candidates[0].is_recommended is True

    def test_should_drop_engines_not_in_model_compatible_engines(self):
        model = _whisper_model(compatible_engines=("transformers",))

        candidates = rank_engines(
            model, _CUDA_PROFILE, [_TRANSFORMERS, _FASTER_WHISPER],
        )

        engine_ids = {c.engine_id for c in candidates}
        assert engine_ids == {"transformers"}

    def test_should_apply_per_model_priority_override(self):
        model = _whisper_model(
            engine_priorities={"transformers": 999, "faster_whisper": 1},
        )

        candidates = rank_engines(
            model, _CUDA_PROFILE, [_TRANSFORMERS, _FASTER_WHISPER],
        )

        assert candidates[0].engine_id == "transformers"
        assert candidates[0].score == 999
        assert candidates[1].engine_id == "faster_whisper"
        assert candidates[1].is_recommended is False

    def test_should_drop_engines_not_supported_by_hardware(self):
        cuda_only = EngineDescriptor(
            engine_id="faster_whisper",
            display_name="faster-whisper (CTranslate2)",
            supported_hardware=("nvidia_cuda",),
            base_priority=80,
        )
        model = _whisper_model()

        candidates = rank_engines(
            model, _ROCM_PROFILE, [_TRANSFORMERS, cuda_only],
        )

        engine_ids = {c.engine_id for c in candidates}
        assert engine_ids == {"transformers"}

    def test_should_return_empty_when_no_engine_is_compatible(self):
        model = _whisper_model(compatible_engines=("nemo",))  # type: ignore[arg-type]

        candidates = rank_engines(
            model, _ROCM_PROFILE, [_TRANSFORMERS, _FASTER_WHISPER],
        )

        assert candidates == ()
