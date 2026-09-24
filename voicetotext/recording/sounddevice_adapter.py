import logging
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from voicetotext.core.exceptions import AudioRecorderError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.recording.input_devices import (
    SYSTEM_DEFAULT,
    input_device_index,
    refresh_portaudio,
)
from voicetotext.recording.recorder import AudioRecorder

_logger = logging.getLogger(__name__)


def _system_default() -> str:
    return SYSTEM_DEFAULT


def _ignore_missing(_name: str) -> None:
    return None


class SoundDeviceAdapter(AudioRecorder):
    """Records from the chosen microphone, or the Windows default without it.

    The choice is read at every start, so picking another microphone takes
    effect on the next recording, and one that was unplugged is used again
    as soon as it is back. ``on_device_missing`` hears about a missing
    microphone once per absence, not once per recording.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device_name: Callable[[], str] = _system_default,
        on_device_missing: Callable[[str], None] = _ignore_missing,
        device_index: Callable[[str], int | None] = input_device_index,
        refresh_devices: Callable[[], None] = refresh_portaudio,
    ) -> None:
        self._sample_rate = sample_rate
        self._device_name = device_name
        self._on_device_missing = on_device_missing
        self._device_index = device_index
        self._refresh_devices = refresh_devices
        self._reported_missing: str | None = None
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._is_recording: bool = False
        self._paused = False
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
            self._paused = False
            self._on_frame = on_frame
            name = self._device_name()
            try:
                self._stream = self._open(self._resolve(name))
            except sd.PortAudioError:
                if name == SYSTEM_DEFAULT:
                    raise
                _logger.warning("Opening microphone %r failed.", name, exc_info=True)
                self._report_missing(name)
                self._stream = self._open(None)
            self._stream.start()
            self._is_recording = True

    def _open(self, device: int | None) -> sd.InputStream:
        return sd.InputStream(
            samplerate=self._sample_rate,
            device=device,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._audio_callback,
        )

    def _resolve(self, name: str) -> int | None:
        if name == SYSTEM_DEFAULT:
            return None
        index = self._device_index(name)
        if index is None:
            # PortAudio only scans at start-up, so a microphone plugged in
            # since is invisible until it looks again.
            self._refresh_devices()
            index = self._device_index(name)
        if index is None:
            self._report_missing(name)
            return None
        self._reported_missing = None
        return index

    def _report_missing(self, name: str) -> None:
        if self._reported_missing == name:
            return
        self._reported_missing = name
        try:
            self._on_device_missing(name)
        except Exception:  # a notice must never stop a recording
            _logger.exception("Reporting a missing microphone failed.")

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

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self.current_rms = 0.0

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
        if self._paused:
            return
        chunk = indata.reshape(-1).copy()
        self.current_rms = float(np.sqrt(np.mean(np.square(chunk))))
        self._chunks.append(chunk)
        on_frame = self._on_frame
        if on_frame is not None:
            on_frame(chunk)
