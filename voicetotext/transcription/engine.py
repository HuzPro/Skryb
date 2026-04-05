from abc import ABC, abstractmethod

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.result import TranscriptionResult


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
