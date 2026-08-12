from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from voicetotext.transcription.result import TranscriptionResult


@dataclass(frozen=True)
class StreamingUpdate:
    """A snapshot of a streaming transcription after one fed chunk.

    ``committed_text`` is locked in and never retracted; ``tentative_text``
    is the current best guess for the trailing audio and may change on the
    next chunk.
    """

    committed_text: str
    tentative_text: str
    is_final: bool

    @property
    def text(self) -> str:
        return " ".join(
            part for part in (self.committed_text, self.tentative_text) if part
        )


class StreamingTranscriptionSession(ABC):
    """One live transcription in progress - fed audio chunk by chunk."""

    @abstractmethod
    def feed(self, chunk: np.ndarray) -> StreamingUpdate:
        """Push the next audio frame (float32, 16 kHz mono) and get an update."""

    @abstractmethod
    def finalize(self) -> TranscriptionResult:
        """Close the stream, commit any remainder, return the full result."""


class StreamingTranscriptionEngine(ABC):
    """Factory for streaming sessions - owns the loaded model."""

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def is_model_loaded(self) -> bool: ...

    @abstractmethod
    def start_session(self) -> StreamingTranscriptionSession: ...

    @abstractmethod
    def dispose(self) -> None: ...
