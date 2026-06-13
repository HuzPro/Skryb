import shutil
from pathlib import Path

from voicetotext.downloads.chunked_file_downloader import ChunkedFileDownloader
from voicetotext.downloads.download_manager import DownloadHandle, DownloadManager
from voicetotext.downloads.hf_repo_resolver import HfRepoResolver
from voicetotext.downloads.model_download_job import ModelDownloadJob
from voicetotext.transcription.model_info import WHISPER_MODELS, ModelInfo
from voicetotext.transcription.model_manager import ModelManager


_TRANSFORMERS_CONFIG_FILENAME = "config.json"


def repo_has_loadable_snapshot(repo_dir: Path) -> bool:
    """Whether *repo_dir* holds a snapshot a ``local_files_only`` load can use.

    HuggingFace's ``snapshot_download(local_files_only=True)`` — the call
    every engine's loader makes under the hood (the transformers Whisper
    adapter, and nano-parakeet via its patched ``hf_hub_download``) —
    resolves the default ``main`` revision through
    ``refs/main`` and then reads ``snapshots/<sha>``. So a snapshot is
    only *loadable* when ``refs/main`` exists AND the snapshot directory it
    names holds a complete model.

    :class:`~voicetotext.downloads.model_download_job.ModelDownloadJob`
    writes ``refs/main`` last, after every file lands, so a partial or
    cancelled download has files on disk but no ``refs/main`` — and is
    therefore correctly reported as not installed. This is the single
    source of truth shared by :meth:`HuggingFaceModelManager.is_installed`
    and :func:`engine_factory.engines_with_model_on_disk`, so the install
    state shown in the UI can never disagree with what the loader can open.
    """
    commit_sha = _resolved_main_revision(repo_dir)
    if commit_sha is None:
        return False
    snapshot = repo_dir / "snapshots" / commit_sha
    return (
        snapshot.is_dir()
        and HuggingFaceModelManager._snapshot_has_complete_model(snapshot)
    )


def _resolved_main_revision(repo_dir: Path) -> str | None:
    refs_main = repo_dir / "refs" / "main"
    if not refs_main.exists():
        return None
    commit_sha = refs_main.read_text().strip()
    return commit_sha or None


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
        return repo_has_loadable_snapshot(self._model_cache_dir(name))

    @staticmethod
    def _snapshot_has_complete_model(snapshot: Path) -> bool:
        return (
            HuggingFaceModelManager._is_nemo_layout(snapshot)
            or HuggingFaceModelManager._is_transformers_layout(snapshot)
        )

    @staticmethod
    def _is_nemo_layout(snapshot: Path) -> bool:
        # NeMo single-file layout (Parakeet): a `.nemo` file alone is a
        # complete model — nano-parakeet doesn't need config.json or
        # safetensors weights.
        return any(p.suffix == ".nemo" for p in snapshot.iterdir() if p.is_file())

    @staticmethod
    def _is_transformers_layout(snapshot: Path) -> bool:
        # Standard transformers layout: ``config.json`` + weights. Weights
        # may be a single ``model.safetensors`` / ``pytorch_model.bin``
        # (Whisper) or a sharded layout indexed by
        # ``model.safetensors.index.json`` with ``model-NNNNN-of-MMMMM``
        # shards (Granite Speech and other multi-GB models).
        if not (snapshot / _TRANSFORMERS_CONFIG_FILENAME).exists():
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
