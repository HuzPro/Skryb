from collections.abc import Callable, Mapping
from pathlib import Path

from voicetotext.downloads.download_manager import DownloadHandle
from voicetotext.transcription.engine_types import EngineId
from voicetotext.transcription.model_info import ModelInfo
from voicetotext.transcription.model_manager import ModelManager


ActiveEngineProvider = Callable[[], EngineId]


class CompositeModelManager(ModelManager):
    """Composite (GoF) over per-engine :class:`ModelManager` children.

    Each engine has its own on-disk cache (``models--openai--whisper-*``
    for transformers, ``models--Systran--faster-whisper-*`` for
    faster-whisper / CTranslate2), so the composite delegates routing
    decisions to the *active* engine — the one currently selected in
    the user's config — for actions whose target storage matters
    (download, delete, get_model_path).

    Listing and installation queries aggregate across children: a model
    counts as installed if *any* engine has its weights on disk, since
    the user can hot-swap into that engine without re-downloading.
    """

    def __init__(
        self,
        by_engine: Mapping[EngineId, ModelManager],
        active_engine_provider: ActiveEngineProvider,
    ) -> None:
        self._by_engine: dict[EngineId, ModelManager] = dict(by_engine)
        self._active_engine_provider = active_engine_provider

    def list_models(self) -> list[ModelInfo]:
        seen: dict[str, ModelInfo] = {}
        for child in self._by_engine.values():
            for info in child.list_models():
                seen.setdefault(info.name, info)
        return list(seen.values())

    def is_installed(self, name: str) -> bool:
        return any(child.is_installed(name) for child in self._by_engine.values())

    def download(self, name: str) -> DownloadHandle:
        return self._active_child().download(name)

    def delete(self, name: str) -> None:
        self._active_child().delete(name)

    def get_model_path(self, name: str) -> Path | None:
        return self._active_child().get_model_path(name)

    def _active_child(self) -> ModelManager:
        return self._by_engine[self._active_engine_provider()]
