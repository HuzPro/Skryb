"""Identity tuple for a single benchmark measurement.

A ``BenchmarkIdentity`` uniquely picks out one "what was measured" in the
JSON history: a model, an engine, the device it ran on, and the audio
fixture it transcribed. The ``PerformanceRecorder`` uses it both as the
record key and as the basis for per-identity monotonic versioning.
"""

from dataclasses import asdict, dataclass
from typing import Literal


DeviceKind = Literal["cpu", "cuda"]
AudioFixtureName = Literal["Testing_1_2_3.mp3", "LongStory.mp3"]


@dataclass(frozen=True, slots=True)
class BenchmarkIdentity:
    model: str
    engine: str
    device: DeviceKind
    audio: AudioFixtureName

    def to_json(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, str]) -> "BenchmarkIdentity":
        return cls(
            model=data["model"],
            engine=data["engine"],
            device=data["device"],  # type: ignore[arg-type]
            audio=data["audio"],  # type: ignore[arg-type]
        )

    def __str__(self) -> str:
        return f"{self.model}/{self.engine}/{self.device}/{self.audio}"
