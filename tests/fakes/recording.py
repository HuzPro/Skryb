from collections.abc import Callable

import numpy as np

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.recorder import AudioRecorder


class FakeAudioRecorder(AudioRecorder):
    """In-memory AudioRecorder.

    Pre-program the segment returned by :meth:`stop_recording` via
    :meth:`will_return`, and failures via :meth:`will_fail_start` or
    :meth:`will_fail_stop`.  Observable state is exposed via plain
    attributes so tests can make direct assertions.
    """

    def __init__(self) -> None:
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.dispose_called: bool = False
        self.captured_on_frame: Callable[[np.ndarray], None] | None = None
        self._is_recording: bool = False
        self._next_segment: AudioSegment | None = None
        self._start_error: Exception | None = None
        self._stop_error: Exception | None = None

    def will_return(self, segment: AudioSegment | None) -> None:
        self._next_segment = segment

    def will_fail_start(self, error: Exception) -> None:
        self._start_error = error

    def will_fail_stop(self, error: Exception) -> None:
        self._stop_error = error

    def start_recording(
        self, on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        self.start_calls += 1
        self.captured_on_frame = on_frame
        if self._start_error is not None:
            raise self._start_error
        self._is_recording = True

    def stop_recording(self) -> AudioSegment | None:
        self.stop_calls += 1
        if self._stop_error is not None:
            raise self._stop_error
        self._is_recording = False
        return self._next_segment

    def is_recording(self) -> bool:
        return self._is_recording

    def dispose(self) -> None:
        self.dispose_called = True
