from collections.abc import Callable, Sequence
from typing import Protocol

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    DeliveryFailedEvent,
    DomainEvent,
    RecordingCancelledEvent,
    TranscriptionFailedEvent,
)
from voicetotext.recording.notice import NoticeAction

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]


class NoticeDisplay(Protocol):
    def show(
        self, message: str, buttons: Sequence[NoticeAction] = (), *, warning: bool = False,
    ) -> None: ...


class RecoveryNotices:
    """Offers the way back right beside a dictation that went wrong.

    Events arrive on whichever thread published them; the notice is shown
    on the GUI thread through ``schedule``. An empty transcription gets no
    notice: there is nothing to recover.
    """

    def __init__(
        self,
        event_bus: EventBus,
        notice: NoticeDisplay,
        schedule: ScheduleOnGuiThread,
        *,
        retry: Callable[[], None],
        open_history: Callable[[], None],
    ) -> None:
        self._notice = notice
        self._schedule = schedule
        self._retry = retry
        self._open_history = open_history
        event_bus.subscribe(TranscriptionFailedEvent, self._on_transcription_failed)
        event_bus.subscribe(DeliveryFailedEvent, self._on_delivery_failed)
        event_bus.subscribe(RecordingCancelledEvent, self._on_recording_cancelled)

    def _on_transcription_failed(self, event: DomainEvent) -> None:
        if not isinstance(event, TranscriptionFailedEvent):
            return
        retry: list[NoticeAction] = [("Retry", self._retry)] if event.recoverable else []
        self._show(
            "Transcription failed.", [*retry, ("Open history", self._open_history)],
            warning=True,
        )

    def _on_delivery_failed(self, event: DomainEvent) -> None:
        if not isinstance(event, DeliveryFailedEvent):
            return
        self._show(
            "No text box here, so it's on your clipboard.",
            [("Open history", self._open_history)],
            warning=True,
        )

    def _on_recording_cancelled(self, event: DomainEvent) -> None:
        if isinstance(event, RecordingCancelledEvent) and event.recoverable:
            self._show("Recording cancelled.", [("Transcribe anyway", self._retry)])

    def _show(
        self, message: str, buttons: list[NoticeAction], *, warning: bool = False,
    ) -> None:
        self._schedule(lambda: self._notice.show(message, buttons, warning=warning))
