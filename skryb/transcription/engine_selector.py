from collections.abc import Sequence

from skryb.transcription.engine_types import (
    EngineCandidate,
    EngineDescriptor,
    HardwareKind,
    HardwareProfile,
)
from skryb.transcription.model_info import ModelInfo


def rank_engines(
    model: ModelInfo,
    hardware: HardwareProfile,
    descriptors: Sequence[EngineDescriptor],
) -> tuple[EngineCandidate, ...]:
    """Strategy: rank the engines that can drive *model* on *hardware*.

    A descriptor survives only when both:
      * its ``engine_id`` is listed in ``model.compatible_engines``, and
      * ``hardware.kind`` is in its ``supported_hardware``.

    Surviving descriptors are scored via the model's per-engine
    priority override (falling back to the descriptor's base priority),
    sorted descending, and the top entry is flagged ``is_recommended``
    with a human-readable reason templated from the hardware kind.
    """
    surviving = _filter_compatible(model, hardware, descriptors)
    if not surviving:
        return ()
    scored = _score(model, surviving)
    ordered = sorted(scored, key=lambda pair: pair[1], reverse=True)
    return _to_candidates(ordered, hardware)


def model_supports_streaming(
    model: ModelInfo,
    descriptors: Sequence[EngineDescriptor],
) -> bool:
    """Whether any engine that can drive *model* supports streaming.

    The descriptor's ``supports_streaming`` flag is the single source of
    truth, so adding a second streaming model later is one descriptor
    flip - no edits here.
    """
    return any(
        d.supports_streaming and d.engine_id in model.compatible_engines
        for d in descriptors
    )


def model_supports_batch(
    model: ModelInfo,
    descriptors: Sequence[EngineDescriptor],
) -> bool:
    """Whether any engine that can drive *model* does batch transcription.

    A model is batch-capable unless every compatible engine is
    ``streaming_only`` (e.g. the cache-aware streaming engine), which lets the
    batch model picker hide streaming-only models.
    """
    compatible = [d for d in descriptors if d.engine_id in model.compatible_engines]
    if not compatible:
        return True  # unknown engine (e.g. tests) - don't hide it
    return any(not d.streaming_only for d in compatible)


def _filter_compatible(
    model: ModelInfo,
    hardware: HardwareProfile,
    descriptors: Sequence[EngineDescriptor],
) -> list[EngineDescriptor]:
    return [
        d for d in descriptors
        if d.engine_id in model.compatible_engines
        and hardware.kind in d.supported_hardware
    ]


def _score(
    model: ModelInfo,
    descriptors: Sequence[EngineDescriptor],
) -> list[tuple[EngineDescriptor, int]]:
    return [
        (d, model.engine_priorities.get(d.engine_id, d.base_priority))
        for d in descriptors
    ]


def _to_candidates(
    ordered: Sequence[tuple[EngineDescriptor, int]],
    hardware: HardwareProfile,
) -> tuple[EngineCandidate, ...]:
    return tuple(
        EngineCandidate(
            engine_id=d.engine_id,
            display_name=d.display_name,
            score=score,
            is_recommended=(idx == 0),
            reason=_reason_for(hardware) if idx == 0 else "",
        )
        for idx, (d, score) in enumerate(ordered)
    )


def _reason_for(hardware: HardwareProfile) -> str:
    label = _HARDWARE_LABELS[hardware.kind]
    if hardware.gfx_arch:
        return f"Recommended for your {label} ({hardware.gfx_arch})"
    return f"Recommended for your {label}"


_HARDWARE_LABELS: dict[HardwareKind, str] = {
    "nvidia_cuda": "NVIDIA CUDA GPU",
    "amd_rocm": "AMD ROCm GPU",
    "cpu": "CPU",
}
