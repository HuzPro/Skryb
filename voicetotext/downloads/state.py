from dataclasses import dataclass
from enum import Enum


class DownloadState(Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(frozen=True)
class ChunkState:
    index: int
    start: int  # inclusive
    end: int    # inclusive (HTTP Range semantics)
    bytes_done: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def remaining(self) -> int:
        return self.length - self.bytes_done

    @property
    def is_complete(self) -> bool:
        return self.bytes_done >= self.length


@dataclass(frozen=True)
class DownloadStats:
    bytes_done: int
    bytes_total: int
    speed_bps: float
    eta_seconds: float | None

    @property
    def fraction(self) -> float:
        if self.bytes_total <= 0:
            return 0.0
        return min(self.bytes_done / self.bytes_total, 1.0)
