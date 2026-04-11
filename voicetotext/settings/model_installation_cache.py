import threading

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
)
from voicetotext.transcription.model_manager import ModelManager


class ModelInstallationCache:
    """Caches ``is_installed`` results so the UI thread never hits the disk.

    The cache is populated lazily (on first ``is_installed`` call) or
    eagerly via :meth:`populate_all` (call at bootstrap from a background
    thread). It subscribes to :class:`DownloadCompletedEvent` and
    :class:`DownloadFailedEvent` to self-invalidate, which covers the
    common cases of downloads finishing while the settings window is open.
    """

    def __init__(self, model_manager: ModelManager, event_bus: EventBus):
        self._model_manager = model_manager
        self._event_bus = event_bus
        self._cache: dict[str, bool] = {}
        self._lock = threading.Lock()

        self._event_bus.subscribe(DownloadCompletedEvent, self._on_download_completed)
        self._event_bus.subscribe(DownloadFailedEvent, self._on_download_failed)

    def is_installed(self, name: str) -> bool:
        with self._lock:
            if name in self._cache:
                return self._cache[name]

        # Fetch without holding the lock — ModelManager.is_installed() hits
        # the filesystem and we don't want to serialize all reads on that.
        result = self._model_manager.is_installed(name)

        with self._lock:
            # Another thread may have won the race; prefer any existing
            # entry so we stay consistent with concurrent invalidations.
            self._cache.setdefault(name, result)
            return self._cache[name]

    def populate_all(self) -> None:
        for info in self._model_manager.list_models():
            installed = self._model_manager.is_installed(info.name)
            with self._lock:
                self._cache[info.name] = installed

    def invalidate(self, name: str) -> None:
        with self._lock:
            self._cache.pop(name, None)

    def dispose(self) -> None:
        self._event_bus.unsubscribe(DownloadCompletedEvent, self._on_download_completed)
        self._event_bus.unsubscribe(DownloadFailedEvent, self._on_download_failed)

    def _on_download_completed(self, event: DownloadCompletedEvent) -> None:
        self.invalidate(event.job_id)

    def _on_download_failed(self, event: DownloadFailedEvent) -> None:
        self.invalidate(event.job_id)
