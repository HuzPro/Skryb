"""Speaker output (WASAPI loopback) and microphone, captured as two 16 kHz mono tracks."""

import collections
import logging
import math
import os
import threading
import time
import warnings
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

RATE = 16000
_GAP = int(0.2 * RATE)  # this far behind the clock means the device went quiet
_MARGIN = int(0.05 * RATE)  # room for a packet still in flight
_POLL_SECONDS = 2.0
_STALL_SECONDS = 5.0
_AUDIBLE_PEAK = 0.01
_ZERO_CROSSINGS = 8
_PASSBAND = 0.92
_LOOPBACK_SUFFIX = " [Loopback]"

_log = logging.getLogger(__name__)

Samples = np.ndarray


class Resampler:
    """Windowed-sinc resampling to 16 kHz that carries its filter state from block to block.

    Output sample j sits at input sample j*rate/16000 exactly, so blocks join
    without a click and an hour of audio does not drift. Output lags input by
    half the filter, under a millisecond.
    """

    def __init__(self, rate: int) -> None:
        self.rate = rate
        common = math.gcd(rate, RATE)
        self._up, self._down = RATE // common, rate // common
        cutoff = 0.5 * min(1.0, RATE / rate) * _PASSBAND  # cycles per input sample
        self._half = math.ceil(_ZERO_CROSSINGS / (2 * cutoff))
        self._taps = np.arange(-self._half + 1, self._half + 1)
        distance = self._taps[None, :] - np.arange(self._up)[:, None] / self._up
        window = 0.5 * (1 + np.cos(np.pi * np.clip(distance / self._half, -1, 1)))
        self._kernels = (2 * cutoff * np.sinc(2 * cutoff * distance) * window).astype(np.float32)
        self._buffer = np.zeros(self._half, np.float32)
        self._first = -self._half  # input index of _buffer[0]
        self._next = 0  # index of the next output sample

    def __call__(self, samples: Samples) -> Samples:
        buffer = np.concatenate([self._buffer, samples.astype(np.float32, copy=False)])
        usable = self._first + len(buffer) - 1 - self._half
        last = ((usable + 1) * self._up - 1) // self._down if usable >= 0 else -1
        outputs = np.arange(self._next, last + 1, dtype=np.int64)
        bases = outputs * self._down // self._up
        phases = outputs * self._down % self._up
        windows = buffer[(bases - self._first)[:, None] + self._taps[None, :]]
        out = np.einsum("ij,ij->i", windows, self._kernels[phases])
        self._next = max(self._next, last + 1)
        drop = max(0, self._next * self._down // self._up - self._half + 1 - self._first)
        self._buffer, self._first = buffer[drop:], self._first + drop
        return out.astype(np.float32)

    def flush(self) -> Samples:
        """What is still held back, with silence for the missing right context."""
        return self(np.zeros(self._half + 1, np.float32))


class Track:
    """One 16 kHz mono stream whose sample N always sits N/16000 s after start().

    WASAPI loopback sends nothing at all while nothing plays, so read() fills
    those gaps with silence from the wall clock instead of letting time collapse.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._blocks: collections.deque[tuple[float, int, int, bytes]] = collections.deque()
        self._t0 = 0.0
        self._emitted = 0
        self._resampler: Resampler | None = None
        self.last_block_at = 0.0

    def start(self) -> None:
        self._t0 = self._clock()
        self.last_block_at = self._t0

    def feed(self, data: bytes, rate: int, channels: int) -> None:
        """Audio-callback side: stash the block; read() does the work."""
        now = self._clock()
        self.last_block_at = now
        self._blocks.append((now, rate, channels, data))

    def read(self) -> Samples:
        out: list[Samples] = []
        while self._blocks:
            arrived, rate, channels, data = self._blocks.popleft()
            mono = np.frombuffer(data, np.float32).reshape(-1, channels).mean(axis=1)
            samples = self._resample(mono, rate)
            self._pad_to(self._position(arrived) - len(samples), out)
            out.append(samples)
            self._emitted += len(samples)
        self._pad_to(self._position(self._clock()) - _MARGIN, out)
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def _resample(self, mono: Samples, rate: int) -> Samples:
        tail = np.zeros(0, np.float32)
        if self._resampler is None or rate != self._resampler.rate:
            if self._resampler is not None:
                tail = self._resampler.flush()
            self._resampler = Resampler(rate)
        return np.concatenate([tail, self._resampler(mono)])

    def _position(self, t: float) -> int:
        return int((t - self._t0) * RATE)

    def _pad_to(self, position: int, out: list[Samples]) -> None:
        if position - self._emitted > _GAP:
            out.append(np.zeros(position - self._emitted, np.float32))
            self._emitted = position


def pick_output(busy: list[str], current: str | None, communications: str | None, default: str | None) -> str | None:
    """The output device to record: stay put while it is still in use, else prefer
    the communications default, then the normal default, then anything busy."""
    if current in busy:
        return current
    for name in (communications, default):
        if name in busy:
            return name
    if busy:
        return busy[0]
    return current or default


class _Stream(Protocol):
    def is_active(self) -> bool: ...
    def stop_stream(self) -> None: ...
    def close(self) -> None: ...


DeviceInfo = dict[str, object]


class Capture:
    """Records speaker output and, when asked, the microphone, keeping the speaker
    side on whichever output device a meeting app is actually playing through."""

    def __init__(self, record_microphone: bool = True) -> None:
        self.them = Track()
        self.me = Track()
        self.output_name: str | None = None
        self.warning: str | None = None
        self._record_microphone = record_microphone
        self._pa: Any = None  # pyaudiowpatch ships no types
        self._loopback: _Stream | None = None
        self._mic: _Stream | None = None
        self._candidate: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import pyaudiowpatch

        self._pa = pyaudiowpatch.PyAudio()
        self.them.start()
        self.me.start()
        try:
            self._open_first_loopback()
        except Exception as exc:
            self._pa.terminate()
            raise RuntimeError("No speaker output found to record.") from exc
        if self._record_microphone and not self._open_mic():
            self.warning = "No microphone found: recording the speakers only."
        self._thread = threading.Thread(target=self._watch, name="meeting-capture-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        for stream in (self._loopback, self._mic):
            _close(stream)
        self._audio().terminate()

    def _open_first_loopback(self) -> None:
        busy, communications, default = _output_state()
        self._open_loopback(pick_output(busy, None, communications, default))

    def _audio(self) -> Any:
        assert self._pa is not None
        return self._pa

    def _open_mic(self) -> bool:
        """The mic is optional: without one, only the speakers are transcribed."""
        _close(self._mic)
        self._mic = None
        try:
            info = self._audio().get_default_wasapi_device(d_in=True)
            self._mic = self._open(info, self.me)
        except OSError:
            return False
        _log.info("Meeting microphone: %s", info["name"])
        return True

    def _open_loopback(self, output_name: str | None) -> None:
        _close(self._loopback)
        info = self._loopback_for(output_name) or self._audio().get_default_wasapi_loopback()
        self._loopback = self._open(info, self.them)
        self.output_name = str(info["name"]).removesuffix(_LOOPBACK_SUFFIX)
        _log.info("Meeting speakers: %s", self.output_name)

    def _loopback_for(self, output_name: str | None) -> DeviceInfo | None:
        for info in self._audio().get_loopback_device_info_generator():
            if str(info["name"]).removesuffix(_LOOPBACK_SUFFIX) == output_name:
                return info
        return None

    def _open(self, info: DeviceInfo, track: Track) -> _Stream:
        import pyaudiowpatch

        rate, channels = int(float(str(info["defaultSampleRate"]))), int(str(info["maxInputChannels"]))

        def callback(data: bytes, frames: int, time_info: object, status: int) -> tuple[None, int]:
            track.feed(data, rate, channels)
            return None, pyaudiowpatch.paContinue

        stream: _Stream = self._audio().open(
            format=pyaudiowpatch.paFloat32, channels=channels, rate=rate, input=True,
            input_device_index=info["index"], frames_per_buffer=rate // 100, stream_callback=callback,
        )
        return stream

    def _watch(self) -> None:
        import comtypes

        comtypes.CoInitialize()
        try:
            while not self._stop.wait(_POLL_SECONDS):
                self._check_streams()
        finally:
            comtypes.CoUninitialize()

    def _check_streams(self) -> None:
        try:
            self._follow_meeting_output()
        except Exception:
            _log.exception("The speaker capture check failed.")
        if self._record_microphone and (self._mic is None or not self._mic.is_active()):
            if self._open_mic():
                _log.info("Meeting microphone (re)connected.")

    def _follow_meeting_output(self) -> None:
        busy, communications, default = _output_state()
        wanted = pick_output(busy, self.output_name, communications, default)
        if self._loopback is None or not self._loopback.is_active():
            _log.warning("The speaker stream stopped; reopening it on %s.", wanted)
            self._open_loopback(wanted)
            self._candidate = None
            return
        if wanted != self.output_name:
            # Switch only when two polls in a row agree, so a blip does not cut audio.
            if self._candidate == wanted:
                self._open_loopback(wanted)
                self._candidate = None
            else:
                self._candidate = wanted
            return
        self._candidate = None
        if self._is_stalled():
            self.warning = f"No sound captured from {self.output_name} while it is playing; reconnected."
            _log.warning(self.warning)
            self._open_loopback(self.output_name)

    def _is_stalled(self) -> bool:
        silent_for = time.monotonic() - self.them.last_block_at
        return silent_for > _STALL_SECONDS and _peak(self.output_name) > _AUDIBLE_PEAK


def _close(stream: _Stream | None) -> None:
    if stream is None:
        return
    try:
        stream.stop_stream()
        stream.close()
    except OSError:
        pass


def _render_devices() -> Any:
    from pycaw.constants import DEVICE_STATE, EDataFlow
    from pycaw.pycaw import AudioUtilities

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pycaw warns on every unreadable device property
        return AudioUtilities.GetAllDevices(EDataFlow.eRender.value, DEVICE_STATE.ACTIVE.value)


def _output_state() -> tuple[list[str], str | None, str | None]:
    """Output devices another process is playing on, plus the communications and normal defaults."""
    from pycaw.api.audiopolicy import IAudioSessionControl2
    from pycaw.constants import EDataFlow, ERole
    from pycaw.pycaw import AudioUtilities

    me = os.getpid()
    busy: list[str] = []
    for device in _render_devices():
        sessions = device.AudioSessionManager.GetSessionEnumerator()
        for i in range(sessions.GetCount()):
            control = sessions.GetSession(i)
            if control is None:
                continue
            session = control.QueryInterface(IAudioSessionControl2)
            if session.GetState() == 1 and session.GetProcessId() != me:
                busy.append(str(device.FriendlyName))
                break
    enumerator = AudioUtilities.GetDeviceEnumerator()
    defaults: list[str | None] = []
    for role in (ERole.eCommunications, ERole.eConsole):
        device = AudioUtilities.CreateDevice(enumerator.GetDefaultAudioEndpoint(EDataFlow.eRender.value, role.value))
        defaults.append(str(device.FriendlyName) if device else None)
    return busy, defaults[0], defaults[1]


def _peak(output_name: str | None) -> float:
    import comtypes
    from pycaw.api.endpointvolume import IAudioMeterInformation

    for device in _render_devices():
        if device.FriendlyName == output_name:
            meter = device._dev.Activate(IAudioMeterInformation._iid_, comtypes.CLSCTX_ALL, None)
            return float(meter.QueryInterface(IAudioMeterInformation).GetPeakValue())
    return 0.0
