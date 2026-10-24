import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainEvent:
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class RecordingStartedEvent(DomainEvent):
    # A few words the pill shows above itself, such as "Editing selection".
    caption: str = ""


@dataclass(frozen=True)
class RecordingStoppedEvent(DomainEvent):
    pass


@dataclass(frozen=True)
class RecordingCappedEvent(DomainEvent):
    max_duration_seconds: float = 0.0


@dataclass(frozen=True)
class RecordingPausedEvent(DomainEvent):
    pass


@dataclass(frozen=True)
class RecordingResumedEvent(DomainEvent):
    pass


@dataclass(frozen=True)
class RecordingCancelledEvent(DomainEvent):
    """``recoverable`` when the discarded audio can still be transcribed."""

    recoverable: bool = False


@dataclass(frozen=True)
class MicrophoneMissingEvent(DomainEvent):
    """The chosen microphone is gone; recording uses the Windows default."""

    name: str = ""


@dataclass(frozen=True)
class MicrophoneSilentEvent(DomainEvent):
    """The recording microphone has sent nothing but digital silence."""

    device_name: str = ""
    seconds_silent: float = 0.0


@dataclass(frozen=True)
class MicrophoneAudioResumedEvent(DomainEvent):
    pass


@dataclass(frozen=True)
class MicrophoneConnectedEvent(DomainEvent):
    """A microphone was plugged in; ``will_be_used`` if the next recording picks it."""

    name: str = ""
    will_be_used: bool = False


@dataclass(frozen=True)
class RecordingCapWarningEvent(DomainEvent):
    remaining_seconds: float = 0.0


@dataclass(frozen=True)
class TranscriptionStartedEvent(DomainEvent):
    """The engine has the audio; ``estimated_seconds`` is None without history."""

    audio_seconds: float = 0.0
    estimated_seconds: float | None = None


@dataclass(frozen=True)
class StateChangedEvent(DomainEvent):
    old_state: str = ""
    new_state: str = ""


@dataclass(frozen=True)
class TextPastedEvent(DomainEvent):
    """Ctrl+V has just gone out with a dictation. Published on the GUI thread."""


@dataclass(frozen=True)
class TranscriptionCompletedEvent(DomainEvent):
    text: str = ""
    transcription_seconds: float = 0.0


@dataclass(frozen=True)
class TranscriptionFailedEvent(DomainEvent):
    """``recoverable`` when the recording can be transcribed again."""

    error: str = ""
    recoverable: bool = False


@dataclass(frozen=True)
class TranscriptionEmptyEvent(DomainEvent):
    """Engine returned no text - surfaces "no speech detected" to the UI.

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


@dataclass(frozen=True)
class ModelDeletedEvent(DomainEvent):
    """Published after a model's files are removed and its cache cleared.

    The settings UI subscribes to this so the affected row's button
    flips from "Delete" back to "Download" without a manual refresh.
    """
    name: str = ""


@dataclass(frozen=True)
class WritingNoticeEvent(DomainEvent):
    """Why a voice edit or cleanup did not happen, shown beside the dictation."""

    message: str = ""


@dataclass(frozen=True)
class WritingServerFailedEvent(DomainEvent):
    """The writing model gave up after its restarts; said once per session."""

    message: str = ""


@dataclass(frozen=True)
class DeliveryFailedEvent(DomainEvent):
    """Dictated text did not reach a text field; ``reason`` says why.

    ``reason`` is a :class:`skryb.output.app_aware_delivery.DeliveryFailure`
    value: ``no_text_field`` (the text went to the clipboard instead).
    """

    reason: str = ""
    text: str = ""


@dataclass(frozen=True)
class NoLastDictationEvent(DomainEvent):
    """Paste-last or copy-last was asked for before anything was dictated."""


@dataclass(frozen=True)
class RetryStartedEvent(DomainEvent):
    """A cancelled or failed recording is being transcribed again."""
