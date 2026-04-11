"""Reusable test fakes for settings-layer unit tests."""

from typing import Any

from voicetotext.core.config import ConfigurationStore
from voicetotext.transcription.model_info import ModelInfo, TRANSFORMERS_WHISPER_MODELS
from voicetotext.transcription.model_manager import ModelManager


class FakeConfig(ConfigurationStore):
    """In-memory config with .save() call counter."""

    def __init__(self, initial: dict[str, Any] | None = None):
        self._data: dict[str, Any] = dict(initial or {})
        self.save_count = 0

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def load(self) -> None:
        pass

    def save(self) -> None:
        self.save_count += 1

    def get_all(self) -> dict:
        return dict(self._data)


class FakeModelManager(ModelManager):
    """ModelManager stub with configurable installed state and call counters."""

    def __init__(
        self,
        installed: dict[str, bool] | None = None,
        models: dict[str, ModelInfo] | None = None,
    ):
        self._installed = dict(installed or {})
        self._models = models or TRANSFORMERS_WHISPER_MODELS
        self.download_calls: list[str] = []
        self.delete_calls: list[str] = []

    def list_models(self) -> list[ModelInfo]:
        return list(self._models.values())

    def is_installed(self, name: str) -> bool:
        return self._installed.get(name, False)

    def download(self, name: str):
        self.download_calls.append(name)

        class _FakeHandle:
            def pause(self_inner) -> None: ...
            def resume(self_inner) -> None: ...
            def cancel(self_inner) -> None: ...

        return _FakeHandle()

    def delete(self, name: str) -> None:
        self.delete_calls.append(name)
        self._installed.pop(name, None)

    def get_model_path(self, name: str):
        return None
