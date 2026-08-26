from pathlib import Path

from voicetotext.downloads.download_manager import DownloadHandle
from voicetotext.transcription.model_info import ALL_MODELS, ModelInfo
from voicetotext.transcription.model_manager import ModelManager


class _FakeDownloadHandle:
    """Minimal DownloadHandle stub - no-op pause/resume/cancel."""

    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def cancel(self) -> None: ...


class FakeModelManager(ModelManager):
    """In-memory :class:`ModelManager` with configurable installed state.

    Records every ``download`` and ``delete`` call so tests can assert
    on them.  Returns a trivial no-op download handle - tests that
    need to assert on download progress should use the download
    coordinator layer, not the manager.
    """

    def __init__(
        self,
        installed: dict[str, bool] | None = None,
        models: dict[str, ModelInfo] | None = None,
    ) -> None:
        self._installed: dict[str, bool] = dict(installed or {})
        self._models = ALL_MODELS if models is None else models
        self.download_calls: list[str] = []
        self.delete_calls: list[str] = []

    def list_models(self) -> list[ModelInfo]:
        return list(self._models.values())

    def has_model(self, name: str) -> bool:
        return name in self._models

    def is_installed(self, name: str) -> bool:
        return self._installed.get(name, False)

    def download(self, name: str) -> DownloadHandle:
        self.download_calls.append(name)
        return _FakeDownloadHandle()  # type: ignore[return-value]

    def delete(self, name: str) -> None:
        self.delete_calls.append(name)
        self._installed.pop(name, None)

    def get_model_path(self, name: str) -> Path | None:
        return None
