from abc import ABC, abstractmethod

import numpy as np

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    RecordingCapWarningEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    StateChangedEvent,
)
from voicetotext.core.state_machine import AppState


SAMPLE_RATE = 16000
_TONE_SECONDS = 0.09
_FADE_SECONDS = 0.005
_AMPLITUDE = 0.2
# A rising "boop" on start, a lower one on stop, so the two are
# distinguishable by ear.
START_FREQUENCY_HZ = 880.0
STOP_FREQUENCY_HZ = 587.33
# A two-note ascending chime for "engine ready / you can dictate now" - longer
# and clearly different from the single start/stop boops.
READY_LOW_FREQUENCY_HZ = 659.25   # E5
READY_HIGH_FREQUENCY_HZ = 987.77  # B5
# The limit is near: two soft, low taps, quieter than the start and stop
# cues, so it reads as a nudge rather than the recording ending.
WARNING_FREQUENCY_HZ = 523.25  # C5
_WARNING_TAP_SECONDS = 0.06
_WARNING_GAP_SECONDS = 0.07
_WARNING_AMPLITUDE = 0.1


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


def warning_to_samples(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Render the soft two-tap 'limit is near' cue as float32 mono samples."""
    tap = tone_to_samples(
        WARNING_FREQUENCY_HZ, _WARNING_TAP_SECONDS, sample_rate, _WARNING_AMPLITUDE,
    )
    gap = np.zeros(int(_WARNING_GAP_SECONDS * sample_rate), dtype=np.float32)
    return np.concatenate([tap, gap, tap])


def ready_chime_to_samples(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Render the two-note ascending 'ready' chime as float32 mono samples."""
    low = tone_to_samples(READY_LOW_FREQUENCY_HZ, sample_rate=sample_rate)
    high = tone_to_samples(READY_HIGH_FREQUENCY_HZ, sample_rate=sample_rate)
    return np.concatenate([low, high])


class RecordingCuePlayer(ABC):
    """Plays the start/stop recording cues."""

    @abstractmethod
    def play_start(self) -> None: ...

    @abstractmethod
    def play_stop(self) -> None: ...

    @abstractmethod
    def play_warning(self) -> None: ...


class CueBeeper:
    """Observer that plays a cue when recording starts and stops.

    Works in both batch and streaming modes because it keys off the
    use-case's recording lifecycle events, not the engine.
    """

    def __init__(self, event_bus: EventBus, player: RecordingCuePlayer) -> None:
        self._player = player
        event_bus.subscribe(RecordingStartedEvent, self._on_started)
        event_bus.subscribe(RecordingStoppedEvent, self._on_stopped)
        event_bus.subscribe(RecordingCapWarningEvent, self._on_cap_warning)

    def _on_started(self, event: RecordingStartedEvent) -> None:
        self._player.play_start()

    def _on_stopped(self, event: RecordingStoppedEvent) -> None:
        self._player.play_stop()

    def _on_cap_warning(self, event: RecordingCapWarningEvent) -> None:
        self._player.play_warning()


class ReadyCuePlayer(ABC):
    """Plays the cue announcing the engine is ready to transcribe."""

    @abstractmethod
    def play_ready(self) -> None: ...


class EngineReadyBeeper:
    """Observer that plays a cue when the engine becomes ready to transcribe.

    Two paths signal readiness, and this listens to both: the initial load
    flips the state machine ``LOADING -> IDLE`` (and publishes no completion
    event), while a model hot-swap publishes :class:`EngineLoadCompletedEvent`
    (and changes no state). Failed loads settle via ``ERROR -> IDLE``, so they
    never ding.
    """

    def __init__(self, event_bus: EventBus, player: ReadyCuePlayer) -> None:
        self._player = player
        event_bus.subscribe(StateChangedEvent, self._on_state_changed)
        event_bus.subscribe(EngineLoadCompletedEvent, self._on_load_completed)

    def _on_state_changed(self, event: StateChangedEvent) -> None:
        if self._is_initial_load_finished(event):
            self._player.play_ready()

    def _is_initial_load_finished(self, event: StateChangedEvent) -> bool:
        return (
            event.old_state == AppState.LOADING.value
            and event.new_state == AppState.IDLE.value
        )

    def _on_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        self._player.play_ready()
