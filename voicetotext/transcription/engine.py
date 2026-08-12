from abc import ABC, abstractmethod

import numpy as np

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult


_WARMUP_SAMPLE_RATE = 16000
# 2 s of silence is enough to traverse the full encoder → decoder
# pipeline on every adapter we ship - feature extraction, GPU
# transfer, kernel compilation, and at least one decoder step - while
# staying short enough that the cost is invisible to the user.
_WARMUP_DURATION_SECONDS = 2.0


class TranscriptionEngine(ABC):

    @abstractmethod
    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        pass

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def is_model_loaded(self) -> bool:
        pass

    @abstractmethod
    def dispose(self) -> None:
        pass

    def warmup(self) -> None:
        """Run one inference pass to compile kernels and warm allocators.

        Pays the per-engine cold-start cost during app startup so the
        user's first real transcription runs at warm-cache speed.
        Adapters may override with a cheaper warmup or a no-op, but
        the default works for every backend we ship.
        """
        if not self.is_model_loaded():
            return
        self.transcribe(_build_silent_warmup_audio())


def _build_silent_warmup_audio() -> AudioSegment:
    samples = int(_WARMUP_SAMPLE_RATE * _WARMUP_DURATION_SECONDS)
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=_WARMUP_SAMPLE_RATE,
    )
