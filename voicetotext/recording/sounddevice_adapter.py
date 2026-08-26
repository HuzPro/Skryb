import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from voicetotext.core.exceptions import AudioRecorderError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.recorder import AudioRecorder


class SoundDeviceAdapter(AudioRecorder):

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._is_recording: bool = False
        self._on_frame: Callable[[np.ndarray], None] | None = None
        self._lock = threading.Lock()
        self.current_rms = 0.0

    def start_recording(
        self, on_frame: Callable[[np.ndarray], None] | None = None,
    ) -> None:
        with self._lock:
            if self._is_recording:
                raise AudioRecorderError("already recording")
            self._chunks = []
            self.current_rms = 0.0
            self._on_frame = on_frame
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._is_recording = True

    def stop_recording(self) -> AudioSegment | None:
        with self._lock:
            if not self._is_recording:
                raise AudioRecorderError("not recording")
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._is_recording = False
            self._on_frame = None
            chunks = self._chunks
            self._chunks = []
            self.current_rms = 0.0
        if not chunks:
            return None
        audio_data = np.concatenate(chunks, axis=0).flatten()
        return AudioSegment(data=audio_data, sample_rate=self._sample_rate)

    def is_recording(self) -> bool:
        return self._is_recording

    def dispose(self) -> None:
        with self._lock:
            self._on_frame = None
            if self._is_recording:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                self._is_recording = False
                self._chunks = []
            self.current_rms = 0.0

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        # sounddevice delivers (frames, channels); for our mono stream that
        # is (frames, 1). Flatten to a 1-D (frames,) mono buffer so every
        # consumer gets the "float32, mono" frame the recorder contract
        # promises. The batch path already flattened at stop; the streaming
        # path fed the raw 2-D frame straight into the model, where
        # ``torch.stft`` reflect-padded the size-1 trailing axis and crashed.
        chunk = indata.reshape(-1).copy()
        self.current_rms = float(np.sqrt(np.mean(np.square(chunk))))
        self._chunks.append(chunk)
        on_frame = self._on_frame
        if on_frame is not None:
            on_frame(chunk)
