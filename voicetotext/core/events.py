import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainEvent:
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RecordingStartedEvent(DomainEvent):
    pass


@dataclass(frozen=True)
class RecordingStoppedEvent(DomainEvent):
    pass


@dataclass(frozen=True)
class StateChangedEvent(DomainEvent):
    old_state: str = ""
    new_state: str = ""


@dataclass(frozen=True)
class TranscriptionCompletedEvent(DomainEvent):
    text: str = ""
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class TranscriptionFailedEvent(DomainEvent):
    error: str = ""


@dataclass(frozen=True)
class ErrorOccurredEvent(DomainEvent):
    error: str = ""
