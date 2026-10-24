import json
import shutil
from pathlib import Path
from typing import cast

from skryb.core.exceptions import ConfigurationError
from skryb.downloads.chunked_file_downloader import ChunkedFileDownloader
from skryb.downloads.download_manager import DownloadHandle, DownloadManager
from skryb.downloads.hf_repo_resolver import HfRepoResolver
from skryb.downloads.model_download_job import ModelDownloadJob
from skryb.transcription.ct2_conversion import prebuild_ct2_model
from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import ModelInfo
from skryb.transcription.model_manager import ModelManager


_CONFIG_FILENAME = "config.json"
_SHARD_INDEX_FILENAME = "model.safetensors.index.json"
_PACKED_ARCHIVE_SUFFIX = ".tar.zst"


def repo_has_loadable_snapshot(repo_dir: Path, revision: str = "") -> bool:
    """Whether *repo_dir* holds a snapshot a ``local_files_only`` load can use.

    HuggingFace's ``snapshot_download(local_files_only=True)`` resolves the
    default ``main`` revision through
    ``refs/main`` and then reads ``snapshots/<sha>``. So a snapshot is
    only *loadable* when ``refs/main`` exists AND the snapshot directory it
    names holds a complete model.

    :class:`~skryb.downloads.model_download_job.ModelDownloadJob`
    writes ``refs/main`` last, after every file lands, so a partial or
    cancelled download has files on disk but no ``refs/main`` - and is
    therefore correctly reported as not installed. This is the single
    source of truth shared by :meth:`HuggingFaceModelManager.is_installed`
    and :func:`engine_factory.engines_with_model_on_disk`, so the install
    state shown in the UI can never disagree with what the loader can open.

    A non-empty *revision* additionally demands that ``refs/main`` name
    that exact commit. Going through ``refs/main`` rather than probing
    ``snapshots/<revision>`` is what keeps the completion barrier intact:
    the snapshot directory exists from the first byte written, so probing
    it would report a half-downloaded pinned model as installed. Equality
    also retires a legacy cache fetched before the pin existed - it sits
    at a different commit and correctly reads as not installed.
    """
    commit_sha = _resolved_main_revision(repo_dir)
    if commit_sha is None:
        return False
    if revision and commit_sha != revision:
        return False
    snapshot = repo_dir / "snapshots" / commit_sha
    return (
        snapshot.is_dir()
        and HuggingFaceModelManager._snapshot_has_complete_model(snapshot)
    )


def loadable_snapshot_dir(
    models_dir: Path, repo_id: str, revision: str = "",
) -> Path:
    """The snapshot directory a ``local_files_only`` load would open.

    Resolves through ``refs/main`` for the same reason
    :func:`repo_has_loadable_snapshot` does: the snapshot directory
    exists from the first byte written, so naming it directly would hand
    back a half-downloaded model.
    """
    repo_dir = models_dir / f"models--{repo_id.replace('/', '--')}"
    commit_sha = _resolved_main_revision(repo_dir)
    if commit_sha is None or (revision and commit_sha != revision):
        raise ConfigurationError(f"{repo_id} is not installed")
    return repo_dir / "snapshots" / commit_sha


def _resolved_main_revision(repo_dir: Path) -> str | None:
    refs_main = repo_dir / "refs" / "main"
    if not refs_main.exists():
        return None
    commit_sha = refs_main.read_text().strip()
    return commit_sha or None


def _every_indexed_shard_present(snapshot: Path) -> bool:
    """Whether *snapshot* holds every weight shard its index names.

    ``model.safetensors.index.json`` is a manifest, not weights: an
    interrupted download leaves the 80 KB index on disk beside none of
    the multi-gigabyte shards it maps to. An index that cannot be read,
    or that names no shard at all, counts as incomplete, so a damaged
    cache is re-downloaded rather than handed to a loader that will
    fail on the missing bytes.
    """
    shards = _indexed_shard_filenames(snapshot / _SHARD_INDEX_FILENAME)
    return bool(shards) and all(
        (snapshot / filename).is_file() for filename in shards
    )


def _indexed_shard_filenames(index_path: Path) -> set[str]:
    try:
        index: object = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return set()
    if not isinstance(index, dict):
        return set()
    weight_map: object = cast(dict[object, object], index).get("weight_map")
    if not isinstance(weight_map, dict):
        return set()
    return {
        filename
        for filename in cast(dict[object, object], weight_map).values()
        if isinstance(filename, str)
    }


class HuggingFaceModelManager(ModelManager):
    """ModelManager backed by the chunked DownloadManager and HfRepoResolver.

    Keeps the existing HuggingFace cache layout
    (``models--org--repo/snapshots/<sha>/...``) so engines can load models
    with ``local_files_only=True`` without knowing how they got on disk.

    ``license_policy`` is required, never defaulted: acquiring weights is
    one of the paths the licence gate exists to cover, and a permissive
    default here would be a bypass.
    """

    def __init__(
        self,
        models_dir: Path,
        download_manager: DownloadManager,
        resolver: HfRepoResolver,
        file_downloader: ChunkedFileDownloader,
        models: dict[str, ModelInfo],
        license_policy: LicensePolicy,
    ) -> None:
        self._models_dir = models_dir
        self._download_manager = download_manager
        self._resolver = resolver
        self._file_downloader = file_downloader
        self._models = models
        self._license_policy = license_policy

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
        return repo_has_loadable_snapshot(
            self._model_cache_dir(name), self._models[name].revision,
        )

    @staticmethod
    def _snapshot_has_complete_model(snapshot: Path) -> bool:
        return (
            HuggingFaceModelManager._is_nemo_layout(snapshot)
            or HuggingFaceModelManager._is_safetensors_layout(snapshot)
            or HuggingFaceModelManager._is_packed_archive_layout(snapshot)
        )

    @staticmethod
    def _is_packed_archive_layout(snapshot: Path) -> bool:
        # Phonon-2: the whole model is one zstd tar, unpacked at first load.
        return any(
            p.name.endswith(_PACKED_ARCHIVE_SUFFIX)
            for p in snapshot.iterdir() if p.is_file()
        )

    @staticmethod
    def _is_nemo_layout(snapshot: Path) -> bool:
        # NeMo single-file layout (Parakeet): a `.nemo` file alone is a
        # complete model - nano-parakeet doesn't need config.json or
        # safetensors weights.
        return any(p.suffix == ".nemo" for p in snapshot.iterdir() if p.is_file())

    @staticmethod
    def _is_safetensors_layout(snapshot: Path) -> bool:
        if not (snapshot / _CONFIG_FILENAME).exists():
            return False
        if (snapshot / _SHARD_INDEX_FILENAME).exists():
            return _every_indexed_shard_present(snapshot)
        return (
            (snapshot / "model.safetensors").exists()
            or (snapshot / "pytorch_model.bin").exists()
        )

    # ------------------------------------------------------------------
    # Downloading
    # ------------------------------------------------------------------

    def download(self, name: str) -> DownloadHandle:
        info = self._models[name]
        self._license_policy.assert_accepted(info)
        resolved = self._resolver.resolve(info.repo_id, info.revision)
        job = ModelDownloadJob(
            job_id=name,
            repo_id=info.repo_id,
            commit_sha=resolved.canonical_commit,
            files=resolved.files,
            models_dir=self._models_dir,
            file_downloader=self._file_downloader,
            post_install=prebuild_ct2_model,
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
