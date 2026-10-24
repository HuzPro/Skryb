"""Notices a microphone that records nothing but digital silence.

A muted or dead microphone still opens and still delivers frames, only
every one of them is zero, so a whole dictation can be lost before the
empty transcript gives it away. The watchdog says so while there is still
time to fix it.
"""

import threading
import time
from collections.abc import Callable

from skryb.core.event_bus import EventBus
from skryb.core.events import (
    DomainEvent,
    MicrophoneAudioResumedEvent,
    MicrophoneSilentEvent,
    RecordingPausedEvent,
    RecordingResumedEvent,
    StateChangedEvent,
)
from skryb.core.state_machine import AppState

# Real room noise sits orders of magnitude above this; only a microphone
# that sends exact or near-exact zeros stays under it.
SILENCE_FLOOR = 1e-5
SAMPLE_SECONDS = 0.25
# A microphone that never sent anything is suspect sooner than one that
# has gone quiet mid-sentence, which may just be someone thinking.
SILENT_FROM_START_SECONDS = 4.0
SILENT_AFTER_AUDIO_SECONDS = 6.0

# Calls the action every so often until the returned stop is called.
Ticker = Callable[[float, Callable[[], None]], Callable[[], None]]


def run_every(seconds: float, action: Callable[[], None]) -> Callable[[], None]:
    stopped = threading.Event()

    def run() -> None:
        while not stopped.wait(seconds):
            action()

    threading.Thread(target=run, daemon=True, name="silence-watchdog").start()
    return stopped.set


def _always() -> bool:
    return True


class SilenceWatchdog:
    """Publishes :class:`MicrophoneSilentEvent` once per silent stretch.

    Paused time does not count: a paused recorder reports silence on
    purpose.
    """

    def __init__(
        self,
        event_bus: EventBus,
        level: Callable[[], float],
        device_name: Callable[[], str],
        enabled: Callable[[], bool] = _always,
        clock: Callable[[], float] = time.monotonic,
        ticker: Ticker = run_every,
    ) -> None:
        self._event_bus = event_bus
        self._level = level
        self._device_name = device_name
        self._enabled = enabled
        self._clock = clock
        self._ticker = ticker
        self._lock = threading.Lock()
        self._stop: Callable[[], None] | None = None
        self._watching = False
        self._paused = False
        self._heard_audio = False
        self._silent_for = 0.0
        self._last_sample = 0.0
        self._warned = False
        event_bus.subscribe(StateChangedEvent, self._on_state_changed)
        event_bus.subscribe(RecordingPausedEvent, self._on_paused)
        event_bus.subscribe(RecordingResumedEvent, self._on_resumed)

    def _on_state_changed(self, event: DomainEvent) -> None:
        if not isinstance(event, StateChangedEvent):
            return
        if event.new_state == AppState.RECORDING.value:
            self._start()
        elif event.old_state == AppState.RECORDING.value:
            self._stop_watching()

    def _start(self) -> None:
        self._stop_watching()
        if not self._enabled():
            return
        with self._lock:
            self._watching = True
            self._paused = False
            self._heard_audio = False
            self._silent_for = 0.0
            self._last_sample = self._clock()
            self._warned = False
        self._stop = self._ticker(SAMPLE_SECONDS, self.sample)

    def _stop_watching(self) -> None:
        with self._lock:
            self._watching = False
        stop, self._stop = self._stop, None
        if stop is not None:
            stop()

    def _on_paused(self, _event: DomainEvent) -> None:
        with self._lock:
            self._paused = True

    def _on_resumed(self, _event: DomainEvent) -> None:
        with self._lock:
            self._paused = False
            self._last_sample = self._clock()

    def sample(self) -> None:
        with self._lock:
            resumed, silent_for = self._take_sample()
        if resumed:
            self._event_bus.publish(MicrophoneAudioResumedEvent())
        elif silent_for is not None:
            self._event_bus.publish(MicrophoneSilentEvent(
                device_name=self._device_name(), seconds_silent=silent_for,
            ))

    def _take_sample(self) -> tuple[bool, float | None]:
        """Whether audio just came back, and how long it has been silent if too long."""
        now = self._clock()
        elapsed, self._last_sample = now - self._last_sample, now
        if not self._watching or self._paused:
            return False, None
        if self._level() >= SILENCE_FLOOR:
            self._heard_audio = True
            self._silent_for = 0.0
            resumed, self._warned = self._warned, False
            return resumed, None
        self._silent_for += max(elapsed, 0.0)
        limit = SILENT_AFTER_AUDIO_SECONDS if self._heard_audio else SILENT_FROM_START_SECONDS
        if self._warned or self._silent_for < limit:
            return False, None
        self._warned = True
        return False, self._silent_for
