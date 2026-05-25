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
class RecordingCappedEvent(DomainEvent):
    max_duration_seconds: float = 0.0


@dataclass(frozen=True)
class StateChangedEvent(DomainEvent):
    old_state: str = ""
    new_state: str = ""


@dataclass(frozen=True)
class TranscriptionCompletedEvent(DomainEvent):
    text: str = ""
    transcription_seconds: float = 0.0


@dataclass(frozen=True)
class TranscriptionFailedEvent(DomainEvent):
    error: str = ""


@dataclass(frozen=True)
class TranscriptionEmptyEvent(DomainEvent):
    """Engine returned no text — surfaces "no speech detected" to the UI.

    Without this event the use case silently dropped empty results,
    leaving the user with an empty clipboard and no indication of
    whether the engine ran at all. The tray subscribes to this and
    shows a notification balloon.
    """
    transcription_seconds: float = 0.0


@dataclass(frozen=True)
class ErrorOccurredEvent(DomainEvent):
    error: str = ""


@dataclass(frozen=True)
class DownloadProgressEvent(DomainEvent):
    job_id: str = ""
    bytes_done: int = 0
    bytes_total: int = 0
    speed_bps: float = 0.0
    eta_seconds: float | None = None


@dataclass(frozen=True)
class DownloadStateChangedEvent(DomainEvent):
    job_id: str = ""
    state: str = ""


@dataclass(frozen=True)
class DownloadCompletedEvent(DomainEvent):
    job_id: str = ""


@dataclass(frozen=True)
class DownloadFailedEvent(DomainEvent):
    job_id: str = ""
    error: str = ""


@dataclass(frozen=True)
class EngineLoadStartedEvent(DomainEvent):
    model_name: str = ""


@dataclass(frozen=True)
class EngineLoadCompletedEvent(DomainEvent):
    model_name: str = ""


@dataclass(frozen=True)
class EngineLoadFailedEvent(DomainEvent):
    model_name: str = ""
    error: str = ""
