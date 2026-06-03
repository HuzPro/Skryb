from abc import ABC, abstractmethod

import numpy as np

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    RecordingStartedEvent,
    RecordingStoppedEvent,
)


SAMPLE_RATE = 16000
_TONE_SECONDS = 0.09
_FADE_SECONDS = 0.005
_AMPLITUDE = 0.2
# A rising "boop" on start, a lower one on stop, so the two are
# distinguishable by ear.
START_FREQUENCY_HZ = 880.0
STOP_FREQUENCY_HZ = 587.33


def tone_to_samples(
    frequency_hz: float,
    seconds: float = _TONE_SECONDS,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = _AMPLITUDE,
) -> np.ndarray:
    """Render a short, click-free sine tone as float32 mono samples."""
    count = int(seconds * sample_rate)
    time_axis = np.arange(count, dtype=np.float32) / sample_rate
    wave = amplitude * np.sin(2.0 * np.pi * frequency_hz * time_axis)
    return (wave * _envelope(count, sample_rate)).astype(np.float32)


def _envelope(count: int, sample_rate: int) -> np.ndarray:
    """Linear fade in/out so the tone doesn't click at the edges."""
    envelope = np.ones(count, dtype=np.float32)
    fade = min(count // 2, int(_FADE_SECONDS * sample_rate))
    if fade <= 0:
        return envelope
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    envelope[:fade] = ramp
    envelope[-fade:] = ramp[::-1]
    return envelope


class RecordingCuePlayer(ABC):
    """Plays the start/stop recording cues."""

    @abstractmethod
    def play_start(self) -> None: ...

    @abstractmethod
    def play_stop(self) -> None: ...


class CueBeeper:
    """Observer that plays a cue when recording starts and stops.

    Works in both batch and streaming modes because it keys off the
    use-case's recording lifecycle events, not the engine.
    """

    def __init__(self, event_bus: EventBus, player: RecordingCuePlayer) -> None:
        self._player = player
        event_bus.subscribe(RecordingStartedEvent, self._on_started)
        event_bus.subscribe(RecordingStoppedEvent, self._on_stopped)

    def _on_started(self, event: RecordingStartedEvent) -> None:
        self._player.play_start()

    def _on_stopped(self, event: RecordingStoppedEvent) -> None:
        self._player.play_stop()
