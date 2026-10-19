from dataclasses import dataclass
from typing import Literal, get_args

EngineId = Literal[
    "crisperwhisper", "crisperwhisper_ct2", "nano_parakeet",
    "nemotron_streaming", "phonon2_cpu",
]
HardwareKind = Literal["nvidia_cuda", "amd_rocm", "cpu"]
ALL_HARDWARE_KINDS: frozenset[HardwareKind] = frozenset(get_args(HardwareKind))
CrisperWhisperMode = Literal["intended", "verbatim"]


@dataclass(frozen=True)
class HardwareProfile:
    kind: HardwareKind
    device_name: str
    gfx_arch: str | None


@dataclass(frozen=True)
class EngineDescriptor:
    engine_id: EngineId
    display_name: str
    supported_hardware: tuple[HardwareKind, ...]
    base_priority: int
    supports_streaming: bool = False
    # True for engines that ONLY do live streaming (no batch transcription),
    # so the batch model picker can hide their models.
    streaming_only: bool = False
    # Declared, never probed: startup reads this to decide whether to import
    # torch at all, and importing torch to find out would defeat the point.
    # Defaults to True so an engine that skips the import must say so.
    requires_torch: bool = True


@dataclass(frozen=True)
class EngineCandidate:
    engine_id: EngineId
    display_name: str
    score: int
    is_recommended: bool
    reason: str
