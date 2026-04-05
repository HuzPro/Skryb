from abc import ABC, abstractmethod

from voicetotext.recording.audio_segment import AudioSegment


class AudioRecorder(ABC):

    @abstractmethod
    def start_recording(self) -> None:
        pass

    @abstractmethod
    def stop_recording(self) -> AudioSegment:
        pass

    @abstractmethod
    def is_recording(self) -> bool:
        pass

    @abstractmethod
    def dispose(self) -> None:
        pass
