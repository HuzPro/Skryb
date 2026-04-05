import threading
from enum import Enum

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import StateChangedEvent


class AppState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    ERROR = "error"


_VALID_TRANSITIONS: dict[AppState, set[AppState]] = {
    AppState.IDLE: {AppState.RECORDING, AppState.ERROR},
    AppState.RECORDING: {AppState.TRANSCRIBING, AppState.IDLE, AppState.ERROR},
    AppState.TRANSCRIBING: {AppState.IDLE, AppState.ERROR},
    AppState.ERROR: {AppState.IDLE},
}


class AppStateMachine:

    def __init__(self, event_bus: EventBus):
        self._state = AppState.IDLE
        self._event_bus = event_bus
        self._lock = threading.Lock()

    @property
    def state(self) -> AppState:
        return self._state

    def transition_to(self, new_state: AppState) -> None:
        with self._lock:
            allowed = _VALID_TRANSITIONS.get(self._state, set())
            if new_state not in allowed:
                raise ValueError(
                    f"Invalid transition: {self._state.value} -> {new_state.value}"
                )
            old_state = self._state
            self._state = new_state

        self._event_bus.publish(
            StateChangedEvent(
                old_state=old_state.value,
                new_state=new_state.value,
            )
        )
