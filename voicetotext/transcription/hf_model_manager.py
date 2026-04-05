import shutil
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download
from tqdm import tqdm as _tqdm_base

from voicetotext.transcription.model_info import AVAILABLE_MODELS, ModelInfo
from voicetotext.transcription.model_manager import ModelManager


def _make_progress_tqdm(callback: Callable[[float], None]):
    class _Bridge(_tqdm_base):
        def update(self, n=1):
            super().update(n)
            if self.total and self.total > 0:
                callback(min(self.n / self.total, 1.0))
    return _Bridge


class HuggingFaceModelManager(ModelManager):

    def __init__(self, models_dir: Path):
        self._models_dir = models_dir

    def list_models(self) -> list[ModelInfo]:
        return list(AVAILABLE_MODELS.values())

    def is_installed(self, name: str) -> bool:
        model_dir = self._model_cache_dir(name)
        return model_dir.exists() and any(model_dir.iterdir())

    def download(self, name: str, on_progress: Callable[[float], None]) -> None:
        info = AVAILABLE_MODELS[name]
        snapshot_download(
            info.repo_id,
            cache_dir=str(self._models_dir),
            tqdm_class=_make_progress_tqdm(on_progress),
        )
        on_progress(1.0)

    def delete(self, name: str) -> None:
        model_dir = self._model_cache_dir(name)
        if model_dir.exists():
            shutil.rmtree(model_dir)

    def get_model_path(self, name: str) -> Path | None:
        model_dir = self._model_cache_dir(name)
        if model_dir.exists() and any(model_dir.iterdir()):
            return model_dir
        return None

    def _model_cache_dir(self, name: str) -> Path:
        info = AVAILABLE_MODELS[name]
        repo_dir_name = info.repo_id.replace("/", "--")
        return self._models_dir / f"models--{repo_dir_name}"
