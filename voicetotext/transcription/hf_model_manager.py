import shutil
from pathlib import Path

from voicetotext.downloads.chunked_file_downloader import ChunkedFileDownloader
from voicetotext.downloads.download_manager import DownloadHandle, DownloadManager
from voicetotext.downloads.hf_repo_resolver import HfRepoResolver
from voicetotext.downloads.model_download_job import ModelDownloadJob
from voicetotext.transcription.model_info import WHISPER_MODELS, ModelInfo
from voicetotext.transcription.model_manager import ModelManager


class HuggingFaceModelManager(ModelManager):
    """ModelManager backed by the chunked DownloadManager and HfRepoResolver.

    Keeps the existing HuggingFace cache layout
    (``models--org--repo/snapshots/<sha>/...``) so the
    :class:`TransformersWhisperAdapter` can continue loading models with
    ``local_files_only=True`` without any knowledge of how they got on disk.
    """

    def __init__(
        self,
        models_dir: Path,
        download_manager: DownloadManager,
        resolver: HfRepoResolver,
        file_downloader: ChunkedFileDownloader,
        models: dict[str, ModelInfo] | None = None,
    ) -> None:
        self._models_dir = models_dir
        self._download_manager = download_manager
        self._resolver = resolver
        self._file_downloader = file_downloader
        self._models = models or WHISPER_MODELS

    # ------------------------------------------------------------------
    # Listing / inspection
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        return list(self._models.values())

    def has_model(self, name: str) -> bool:
        return name in self._models

    def is_installed(self, name: str) -> bool:
        if name not in self._models:
            return False
        snapshots_dir = self._model_cache_dir(name) / "snapshots"
        if not snapshots_dir.exists():
            return False
        return any(
            self._snapshot_has_complete_model(snapshot)
            for snapshot in snapshots_dir.iterdir()
        )

    @staticmethod
    def _snapshot_has_complete_model(snapshot: Path) -> bool:
        # NeMo single-file layout (Parakeet): a `.nemo` file alone is a
        # complete model — nano-parakeet doesn't need config.json or
        # safetensors weights.
        if any(p.suffix == ".nemo" for p in snapshot.iterdir() if p.is_file()):
            return True
        # Standard transformers / faster-whisper layout: config + weights.
        # Weights may be a single ``model.safetensors`` /
        # ``pytorch_model.bin`` (Whisper) or a sharded layout indexed by
        # ``model.safetensors.index.json`` with ``model-NNNNN-of-MMMMM``
        # shards (Granite Speech and other multi-GB models).
        if not (snapshot / "config.json").exists():
            return False
        return (
            (snapshot / "model.safetensors").exists()
            or (snapshot / "pytorch_model.bin").exists()
            or (snapshot / "model.safetensors.index.json").exists()
        )

    # ------------------------------------------------------------------
    # Downloading
    # ------------------------------------------------------------------

    def download(self, name: str) -> DownloadHandle:
        info = self._models[name]
        commit_sha, files = self._resolver.resolve(info.repo_id)
        job = ModelDownloadJob(
            job_id=name,
            repo_id=info.repo_id,
            commit_sha=commit_sha,
            files=files,
            models_dir=self._models_dir,
            file_downloader=self._file_downloader,
        )
        return self._download_manager.enqueue(job)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete(self, name: str) -> None:
        model_dir = self._model_cache_dir(name)
        if model_dir.exists():
            shutil.rmtree(model_dir)

    def get_model_path(self, name: str) -> Path | None:
        if self.is_installed(name):
            return self._model_cache_dir(name)
        return None

    def _model_cache_dir(self, name: str) -> Path:
        info = self._models[name]
        repo_dir_name = info.repo_id.replace("/", "--")
        return self._models_dir / f"models--{repo_dir_name}"
