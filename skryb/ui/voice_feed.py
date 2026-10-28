"""Streams the voice level to the UI shell while recording, for its moving logo."""

import threading
from collections.abc import Callable

from skryb.core.event_bus import EventBus
from skryb.core.events import DomainEvent, StateChangedEvent
from skryb.core.state_machine import AppState
from skryb.recording.pill import VoiceLevel
from skryb.recording.silence_watchdog import Ticker, run_every

# About twenty readings a second, the same rate as the microphone meter.
SAMPLE_SECONDS = 0.05


class VoiceFeed:
    """Publishes ``voice`` events of 0..1 loudness while a recording runs.

    The level is scaled between the room's noise floor and the speaker's
    recent peak, the way the pill's waveform is, so the shell's bars move
    the same as the pill's.
    """

    def __init__(
        self,
        event_bus: EventBus,
        publish: Callable[[str, object], None],
        level: Callable[[], float],
        ticker: Ticker = run_every,
    ) -> None:
        self._publish = publish
        self._level = level
        self._ticker = ticker
        self._lock = threading.Lock()
        self._stop: Callable[[], None] | None = None
        self._voice = VoiceLevel()
        event_bus.subscribe(StateChangedEvent, self._on_state_changed)

    def _on_state_changed(self, event: DomainEvent) -> None:
        if not isinstance(event, StateChangedEvent):
            return
        if event.new_state == AppState.RECORDING.value:
            self._start()
        elif event.old_state == AppState.RECORDING.value:
            self._stop_sampling()

    def _start(self) -> None:
        self._stop_sampling()
        with self._lock:
            self._voice = VoiceLevel()
            self._stop = self._ticker(SAMPLE_SECONDS, self.sample)

    def _stop_sampling(self) -> None:
        with self._lock:
            stop, self._stop = self._stop, None
        if stop is not None:
            stop()
            self._publish("voice", {"level": 0.0})

    def sample(self) -> None:
        level = self._voice.update(self._level(), SAMPLE_SECONDS)
        self._publish("voice", {"level": round(level, 3)})
