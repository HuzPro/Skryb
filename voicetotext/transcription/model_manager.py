from abc import ABC, abstractmethod
from pathlib import Path

from voicetotext.downloads.download_manager import DownloadHandle
from voicetotext.transcription.model_info import ModelInfo


class ModelManager(ABC):

    @abstractmethod
    def list_models(self) -> list[ModelInfo]:
        pass

    @abstractmethod
    def is_installed(self, name: str) -> bool:
        pass

    @abstractmethod
    def download(self, name: str) -> DownloadHandle:
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        pass

    @abstractmethod
    def get_model_path(self, name: str) -> Path | None:
        pass
