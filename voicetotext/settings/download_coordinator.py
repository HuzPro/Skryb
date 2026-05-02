import logging

from voicetotext.downloads.download_manager import DownloadHandle
from voicetotext.transcription.model_manager import ModelManager


_logger = logging.getLogger(__name__)


class DownloadCoordinator:
    """Owns the mapping of model name → active :class:`DownloadHandle`.

    The settings UI never touches ``DownloadHandle`` directly; it issues
    commands by name and this coordinator routes them to the correct
    handle. Keeping handle lifetime here means the UI tabs can be rebuilt
    (e.g. first-run → normal mode) without losing in-flight downloads.
    """

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager
        self._handles: dict[str, DownloadHandle] = {}

    def start(self, name: str) -> None:
        _logger.info("download.start: '%s'", name)
        self._handles[name] = self._model_manager.download(name)
        _logger.info("download.start: '%s' enqueued", name)

    def pause(self, name: str) -> None:
        handle = self._require_handle(name)
        handle.pause()

    def resume(self, name: str) -> None:
        # Resume semantics mirror the DownloadManager: re-enqueue the job
        # and let the chunked downloader pick up from the sidecar on disk.
        self.start(name)

    def cancel(self, name: str) -> None:
        handle = self._require_handle(name)
        handle.cancel()
        del self._handles[name]

    def _require_handle(self, name: str) -> DownloadHandle:
        handle = self._handles.get(name)
        if handle is None:
            raise LookupError(f"no active download for model '{name}'")
        return handle
