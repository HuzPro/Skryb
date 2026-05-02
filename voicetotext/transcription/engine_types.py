from dataclasses import dataclass
from typing import Literal


EngineId = Literal["transformers", "faster_whisper", "nano_parakeet"]
HardwareKind = Literal["nvidia_cuda", "amd_rocm", "cpu"]


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


@dataclass(frozen=True)
class EngineCandidate:
    engine_id: EngineId
    display_name: str
    score: int
    is_recommended: bool
    reason: str
