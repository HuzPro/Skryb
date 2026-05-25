from PySide6.QtCore import QObject, Signal

from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    ModelDeletedEvent,
)
from voicetotext.settings.service import DownloadEventListener


class QtDownloadBridge(QObject):
    """Observer that re-emits download events as Qt signals.

    Subscribed to :class:`SettingsService` as a
    :class:`DownloadEventListener` (registered below). Qt's
    AutoConnection marshals the emitted signals onto the GUI thread
    when the receiver is a QObject in that thread, replacing the old
    ``window.after(0, ...)`` pattern.

    Inheritance is single (``QObject`` only) because ``shiboken``'s
    metaclass conflicts with :class:`abc.ABCMeta`; we register as a
    virtual subclass via :meth:`ABC.register` at module import time.

    Also relays :class:`ModelDeletedEvent` even though it isn't part of
    the download lifecycle — the bridge already owns Qt-thread
    affinity for row-refresh actions, so colocating the relay keeps
    the window's wiring symmetric. The bridge does *not* implement an
    extra ABC for this; the :class:`DownloadEventListener` interface
    stays segregated.
    """

    progress = Signal(object)
    state = Signal(object)
    completed = Signal(object)
    failed = Signal(object)
    deleted = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    def on_download_progress(self, event: DownloadProgressEvent) -> None:
        self.progress.emit(event)

    def on_download_state(self, event: DownloadStateChangedEvent) -> None:
        self.state.emit(event)

    def on_download_completed(self, event: DownloadCompletedEvent) -> None:
        self.completed.emit(event)

    def on_download_failed(self, event: DownloadFailedEvent) -> None:
        self.failed.emit(event)

    def on_model_deleted(self, event: ModelDeletedEvent) -> None:
        self.deleted.emit(event)


DownloadEventListener.register(QtDownloadBridge)
