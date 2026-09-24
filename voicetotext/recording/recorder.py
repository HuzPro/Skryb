from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from voicetotext.recording.audio_segment import AudioSegment


class AudioRecorder(ABC):

    @abstractmethod
    def start_recording(
        self, on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        """Begin capturing. If *on_frame* is given, each live audio frame
        (float32, mono) is forwarded to it as it arrives - for streaming."""

    @abstractmethod
    def stop_recording(self) -> AudioSegment | None:
        pass

    @abstractmethod
    def is_recording(self) -> bool:
        pass

    @abstractmethod
    def set_paused(self, paused: bool) -> None:
        """Drop incoming audio while paused, keeping the stream open."""

    @abstractmethod
    def dispose(self) -> None:
        pass
