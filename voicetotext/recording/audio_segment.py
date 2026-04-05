from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AudioSegment:
    data: np.ndarray
    sample_rate: int

    def __post_init__(self):
        if self.data.size == 0:
            raise ValueError("Audio data must not be empty")
        if self.data.dtype != np.float32:
            raise ValueError(f"Audio data must be float32, got {self.data.dtype}")
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    @property
    def sample_count(self) -> int:
        return self.data.shape[0]
