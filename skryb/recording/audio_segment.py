import wave
from dataclasses import dataclass
from pathlib import Path

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


def load_wav(path: Path) -> AudioSegment:
    """Read a 16-bit PCM recording, as the diagnostics logs save them, as mono."""
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2:
            raise wave.Error("only 16-bit recordings can be read")
        channels = source.getnchannels()
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").reshape(-1, channels)
    mono = samples.astype(np.float32).mean(axis=1) / 32768.0
    return AudioSegment(data=mono.astype(np.float32), sample_rate=rate)
