"""Downloading a translation path: its llama.cpp build, then its model, as one job with one progress bar.

The build's archives are checked against their size and SHA-256 and unpacked beside each other, so
a build folder at its final name is always a whole, verified one. The model goes through the app's
Hugging Face downloader into the models folder's usual ``models--org--repo`` layout, pinned to its
commit. A part already on disk is skipped, so a retry after a failure only fetches what is missing.
"""

import hashlib
import shutil
import threading
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from skryb.downloads.chunked_file_downloader import FileDownloadPlan
from skryb.downloads.hf_repo_resolver import ResolvedFile
from skryb.downloads.model_download_job import ModelDownloadJob
from skryb.translation.models import Archive, TranslationModel, TranslationPath

INSTALL_JOB_ID = "translation"
_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_RUNTIMES = "llama.cpp"


class InstallError(Exception):
    """A download that came back wrong; ``str()`` is a sentence to show."""


class FileDownloader(Protocol):
    def download(
        self, plan: FileDownloadPlan, on_progress: Callable[[int], None], cancel_event: threading.Event, pause_event: threading.Event,
    ) -> None: ...


def runtime_exe(models_dir: Path, path: TranslationPath) -> Path:
    return models_dir / _RUNTIMES / path.runtime_name / "llama-server.exe"


def model_file(models_dir: Path, model: TranslationModel) -> Path:
    return models_dir / f"models--{model.repo_id.replace('/', '--')}" / "snapshots" / model.revision / model.file_name


def model_installed(models_dir: Path, model: TranslationModel) -> bool:
    refs_main = models_dir / f"models--{model.repo_id.replace('/', '--')}" / "refs" / "main"
    return _has_size(model_file(models_dir, model), model.size) and _read(refs_main) == model.revision


def installed(models_dir: Path, path: TranslationPath) -> bool:
    return runtime_exe(models_dir, path).is_file() and model_installed(models_dir, path.model)


class InstallJob:
    """Shaped like a download-manager job, so it gets the app's progress events, pause and cancel."""

    def __init__(self, models_dir: Path, path: TranslationPath, downloader: FileDownloader) -> None:
        self._models_dir = models_dir
        self._path = path
        self._downloader = downloader

    @property
    def job_id(self) -> str:
        return INSTALL_JOB_ID

    @property
    def total_bytes(self) -> int:
        return self._path.download_bytes

    def run(self, on_progress: Callable[[int, int], None], cancel_event: threading.Event, pause_event: threading.Event) -> None:
        total, done = self.total_bytes, 0

        def advance(delta: int) -> None:
            nonlocal done
            done += delta
            on_progress(done, total)

        runtime = runtime_exe(self._models_dir, self._path)
        if runtime.is_file():
            advance(sum(archive.size for archive in self._path.runtime))
        else:
            staged = [self._fetch(archive, advance, cancel_event, pause_event) for archive in self._path.runtime]
            _unpack(staged, runtime.parent)
        model = self._path.model
        if model_installed(self._models_dir, model):
            advance(model.size)
        else:
            base = done
            ModelDownloadJob(
                job_id=INSTALL_JOB_ID, repo_id=model.repo_id, commit_sha=model.revision,
                files=[ResolvedFile(model.file_name, model.size, model.url)],
                models_dir=self._models_dir, file_downloader=self._downloader,
            ).run(lambda model_done, _total: on_progress(base + model_done, total), cancel_event, pause_event)
        on_progress(total, total)

    def _fetch(
        self, archive: Archive, advance: Callable[[int], None], cancel_event: threading.Event, pause_event: threading.Event,
    ) -> Path:
        staged = self._models_dir / _RUNTIMES / "downloads" / archive.file_name
        if not _has_size(staged, archive.size):
            plan = FileDownloadPlan(url=archive.url, target=staged, expected_size=archive.size)
            self._downloader.download(plan, advance, cancel_event, pause_event)
        else:
            advance(archive.size)
        if _sha256_of(staged) != archive.sha256:
            staged.unlink(missing_ok=True)
            raise InstallError(f"The download of {archive.file_name} was corrupted; try again.")
        return staged


def _unpack(archives: list[Path], target: Path) -> None:
    """Every archive into one folder, which takes *target*'s name only once all of them are in it."""
    partial = target.with_name(target.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    for archive in archives:
        # ``extractall`` drops absolute paths and ".." parts, so nothing lands outside ``partial``.
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(partial)
    if not (partial / "llama-server.exe").is_file():
        shutil.rmtree(partial, ignore_errors=True)
        raise InstallError("The llama.cpp download has no llama-server.exe in it.")
    shutil.rmtree(target, ignore_errors=True)
    partial.rename(target)
    for archive in archives:
        archive.unlink(missing_ok=True)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _has_size(path: Path, size: int) -> bool:
    try:
        return path.stat().st_size == size
    except OSError:
        return False


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
