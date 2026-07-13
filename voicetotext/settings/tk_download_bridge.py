from collections.abc import Callable
from typing import Protocol

from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    ModelDeletedEvent,
)
from voicetotext.settings.service import DownloadEventListener

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]


class DownloadTarget(Protocol):
    def on_download_progress(self, event: DownloadProgressEvent) -> None: ...
    def on_download_state(self, event: DownloadStateChangedEvent) -> None: ...
    def on_download_completed(self, event: DownloadCompletedEvent) -> None: ...
    def on_download_failed(self, event: DownloadFailedEvent) -> None: ...
    def on_model_deleted(self, event: ModelDeletedEvent) -> None: ...


class TkDownloadBridge(DownloadEventListener):
    """Marshals download lifecycle events onto the Tk GUI thread.

    Subscribed to :class:`SettingsService` as a
    :class:`DownloadEventListener`. Download events fire on the download
    manager's worker threads; ``schedule`` queues each onto the GUI loop
    before invoking the window's slot, replacing Qt's ``AutoConnection``.

    Also relays :class:`ModelDeletedEvent` via :meth:`on_model_deleted`
    (subscribed separately through ``subscribe_model_deleted``); it is
    not part of the :class:`DownloadEventListener` ABC.
    """

    def __init__(self, schedule: ScheduleOnGuiThread, target: DownloadTarget) -> None:
        self._schedule = schedule
        self._target = target

    def on_download_progress(self, event: DownloadProgressEvent) -> None:
        self._schedule(lambda: self._target.on_download_progress(event))

    def on_download_state(self, event: DownloadStateChangedEvent) -> None:
        self._schedule(lambda: self._target.on_download_state(event))

    def on_download_completed(self, event: DownloadCompletedEvent) -> None:
        self._schedule(lambda: self._target.on_download_completed(event))

    def on_download_failed(self, event: DownloadFailedEvent) -> None:
        self._schedule(lambda: self._target.on_download_failed(event))

    def on_model_deleted(self, event: ModelDeletedEvent) -> None:
        self._schedule(lambda: self._target.on_model_deleted(event))
