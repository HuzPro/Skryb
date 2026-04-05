import threading

import numpy as np
import sounddevice as sd

from voicetotext.core.exceptions import AudioRecorderError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.recorder import AudioRecorder


class SoundDeviceAdapter(AudioRecorder):

    def __init__(self, sample_rate: int = 16000):
        self._sample_rate = sample_rate
        self._stream = None
        self._chunks: list[np.ndarray] = []
        self._is_recording = False
        self._lock = threading.Lock()

    def start_recording(self) -> None:
        with self._lock:
            if self._is_recording:
                raise AudioRecorderError("already recording")
            self._chunks = []
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording = True

    def stop_recording(self) -> AudioSegment:
        with self._lock:
            if not self._is_recording:
                raise AudioRecorderError("not recording")
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._is_recording = False
            audio_data = np.concatenate(self._chunks, axis=0).flatten()
            self._chunks = []
        return AudioSegment(data=audio_data, sample_rate=self._sample_rate)

    def is_recording(self) -> bool:
        return self._is_recording

    def dispose(self) -> None:
        with self._lock:
            if self._is_recording:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                self._is_recording = False
                self._chunks = []

    def _audio_callback(self, indata, frames, time_info, status):
        self._chunks.append(indata.copy())
