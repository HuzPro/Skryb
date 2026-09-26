"""Where the writing model lives, and how it gets there.

The managed install sits under ``%LOCALAPPDATA%\\VoiceToText\\writing``: the
pinned llama.cpp build unzipped into ``llama-b11200`` and the pinned GGUF in
``models``. Both are checked against their size and SHA-256 before they are
moved into place, so a file at its final name is always a verified one.
"""

import hashlib
import logging
import os
import shutil
import threading
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
)
from voicetotext.downloads.chunked_file_downloader import FileDownloadPlan
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.writing.llama_server import ServerFiles
from voicetotext.writing.settings import WritingSettings

_logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 4 * 1024 * 1024
INSTALL_JOB_ID = "writing-model"


@dataclass(frozen=True)
class Artifact:
    url: str
    file_name: str
    size: int
    sha256: str


RUNTIME_ZIP = Artifact(
    url=(
        "https://github.com/ggml-org/llama.cpp/releases/download/b11200/"
        "llama-b11200-bin-win-vulkan-x64.zip"
    ),
    file_name="llama-b11200-bin-win-vulkan-x64.zip",
    size=33_062_078,
    sha256="670a1f9bf5272c43dbbd7b0004a12dd3d3f6f575246fd23b733f06c064f1b96b",
)
MODEL_GGUF = Artifact(
    url=(
        "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/"
        "e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf"
    ),
    file_name="Qwen3.5-4B-Q4_K_M.gguf",
    size=2_740_937_888,
    sha256="00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
)
INSTALL_BYTES = RUNTIME_ZIP.size + MODEL_GGUF.size

DOWNLOAD_FAILED = (
    "Could not download the writing model. Check your internet connection and try again."
)
DOWNLOAD_CORRUPTED = "The download was corrupted; try again."
INSTALL_FAILED = "Could not install the writing model. Open the log for details."


class InstallError(Exception):
    """``str()`` is the sentence the Writing page shows; ``detail`` is what went wrong."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail or message


class FileDownloader(Protocol):
    def download(
        self,
        plan: FileDownloadPlan,
        on_progress: Callable[[int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None: ...


class WritingPaths:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def runtime_exe(self) -> Path:
        return self.root / "llama-b11200" / "llama-server.exe"

    @property
    def model_file(self) -> Path:
        return self.root / "models" / MODEL_GGUF.file_name

    @property
    def downloads_dir(self) -> Path:
        return self.root / "downloads"

    @property
    def install_log(self) -> Path:
        return self.root / "install.log"


def managed_root() -> Path:
    local = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(local) / "VoiceToText" / "writing"


def server_files(paths: WritingPaths, settings: WritingSettings) -> ServerFiles | None:
    """The runtime and model to launch, or ``None`` while either is missing.

    An override path wins over the managed file; the managed model must also
    be the full pinned size, so a half-moved file never gets launched.
    """
    runtime = Path(settings.runtime_path) if settings.runtime_path else paths.runtime_exe
    model = Path(settings.model_path) if settings.model_path else paths.model_file
    if not runtime.is_file() or not model.is_file():
        return None
    if not settings.model_path and not _has_size(model, MODEL_GGUF.size):
        return None
    return ServerFiles(runtime=runtime, model=model)


def _has_size(path: Path, size: int) -> bool:
    try:
        return path.stat().st_size == size
    except OSError:
        return False


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


class InstallJob:
    """Downloads, verifies and unpacks the runtime, then the model.

    Shaped like a download-manager job, so it gets the same progress events,
    cancel and resume as an ASR model download. A part already installed is
    skipped, which is what makes a retry after a failure cheap.
    """

    def __init__(
        self,
        paths: WritingPaths,
        downloader: FileDownloader,
        runtime: Artifact = RUNTIME_ZIP,
        model: Artifact = MODEL_GGUF,
    ) -> None:
        self._paths = paths
        self._downloader = downloader
        self._runtime = runtime
        self._model = model

    @property
    def job_id(self) -> str:
        return INSTALL_JOB_ID

    @property
    def total_bytes(self) -> int:
        return self._runtime.size + self._model.size

    def run(
        self,
        on_progress: Callable[[int, int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        """Install, or raise an :class:`InstallError` whose details are in the install log."""
        try:
            self._install(on_progress, cancel_event, pause_event)
        except InstallError as failure:
            self._log_failure(failure.detail, cancel_event)
            raise
        except Exception as exc:
            self._log_failure(repr(exc), cancel_event)
            raise InstallError(INSTALL_FAILED, repr(exc)) from exc

    def _install(
        self,
        on_progress: Callable[[int, int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        total = self.total_bytes
        done = 0

        def advance(delta: int) -> None:
            nonlocal done
            done += delta
            on_progress(done, total)

        if self._paths.runtime_exe.is_file():
            advance(self._runtime.size)
        else:
            archive = self._fetch(self._runtime, advance, cancel_event, pause_event)
            self._unpack_runtime(archive)
        if _has_size(self._paths.model_file, self._model.size):
            advance(self._model.size)
        else:
            staged = self._fetch(self._model, advance, cancel_event, pause_event)
            self._paths.model_file.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, self._paths.model_file)
        on_progress(total, total)

    def _fetch(
        self,
        artifact: Artifact,
        advance: Callable[[int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> Path:
        staged = self._paths.downloads_dir / artifact.file_name
        if not _has_size(staged, artifact.size):
            self._download(artifact, staged, advance, cancel_event, pause_event)
        if cancel_event.is_set():
            raise InstallError("Cancelled")
        if not _has_size(staged, artifact.size):
            staged.unlink(missing_ok=True)
            raise InstallError(
                DOWNLOAD_CORRUPTED, f"{artifact.file_name} is not the expected size.",
            )
        if sha256_of(staged) != artifact.sha256:
            staged.unlink(missing_ok=True)
            raise InstallError(DOWNLOAD_CORRUPTED, f"{artifact.file_name} failed its checksum.")
        return staged

    def _download(
        self,
        artifact: Artifact,
        staged: Path,
        advance: Callable[[int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        plan = FileDownloadPlan(url=artifact.url, target=staged, expected_size=artifact.size)
        try:
            self._downloader.download(plan, advance, cancel_event, pause_event)
        except Exception as exc:
            if cancel_event.is_set() or pause_event.is_set():
                raise
            raise InstallError(
                DOWNLOAD_FAILED, f"Downloading {artifact.url} failed: {exc}",
            ) from exc

    def _log_failure(self, detail: str, cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            return
        _logger.warning("Writing model install failed: %s", detail)
        try:
            _write_install_log(self._paths.install_log, detail)
        except OSError:
            _logger.warning("Could not write %s.", self._paths.install_log, exc_info=True)

    def _unpack_runtime(self, archive: Path) -> None:
        target = self._paths.runtime_exe.parent
        partial = target.with_name(target.name + ".partial")
        shutil.rmtree(partial, ignore_errors=True)
        try:
            _extract(archive, partial)
        except (zipfile.BadZipFile, OSError) as exc:
            shutil.rmtree(partial, ignore_errors=True)
            archive.unlink(missing_ok=True)
            raise InstallError(
                DOWNLOAD_CORRUPTED, f"The runtime archive could not be unpacked: {exc}",
            ) from exc
        if not (partial / self._paths.runtime_exe.name).is_file():
            shutil.rmtree(partial, ignore_errors=True)
            raise InstallError(DOWNLOAD_CORRUPTED, "The runtime archive has no llama-server.exe.")
        shutil.rmtree(target, ignore_errors=True)
        partial.rename(target)
        archive.unlink(missing_ok=True)


def _write_install_log(path: Path, detail: str) -> None:
    """Keep the last failure where the Writing page's Open log button finds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{datetime.now(UTC).isoformat()} install failed: {detail}\n", encoding="utf-8",
    )


def _extract(archive: Path, destination: Path) -> None:
    # ``extractall`` drops absolute paths and ".." parts from member names,
    # so nothing in the archive can land outside ``destination``.
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)


@dataclass(frozen=True)
class InstallProgress:
    state: str = "idle"
    bytes_done: int = 0
    bytes_total: int = 0
    error: str = ""


def chunked_downloader() -> FileDownloader:
    import requests

    from voicetotext.downloads.chunked_file_downloader import ChunkedFileDownloader
    from voicetotext.downloads.requests_http_client import RequestsHttpClient

    return ChunkedFileDownloader(http=RequestsHttpClient(session=requests.Session()))


class WritingInstaller:
    """Runs the install job on its own download manager and bus.

    A private bus keeps these events away from the ASR model list, which
    listens to the app bus by job id.
    """

    def __init__(
        self,
        paths: WritingPaths,
        on_change: Callable[[], None],
        make_downloader: Callable[[], FileDownloader] = chunked_downloader,
        runtime: Artifact = RUNTIME_ZIP,
        model: Artifact = MODEL_GGUF,
    ) -> None:
        self._paths = paths
        self._on_change = on_change
        self._make_downloader = make_downloader
        self._runtime = runtime
        self._model = model
        self._manager: DownloadManager | None = None
        self._progress = InstallProgress()

    @property
    def progress(self) -> InstallProgress:
        return self._progress

    def install(self) -> None:
        if self._manager is None:
            bus = SimpleEventBus()
            bus.subscribe(DownloadProgressEvent, self._on_progress)
            bus.subscribe(DownloadStateChangedEvent, self._on_state)
            bus.subscribe(DownloadCompletedEvent, self._on_completed)
            bus.subscribe(DownloadFailedEvent, self._on_failed)
            self._manager = DownloadManager(bus, max_concurrent_downloads=1)
        self._update(InstallProgress("downloading", 0, self._runtime.size + self._model.size))
        self._manager.enqueue(
            InstallJob(self._paths, self._make_downloader(), self._runtime, self._model),
        )

    def cancel(self) -> None:
        if self._manager is not None:
            self._manager.cancel(INSTALL_JOB_ID)

    def shutdown(self) -> None:
        if self._manager is not None:
            self._manager.shutdown()

    def _on_progress(self, event: DownloadProgressEvent) -> None:
        self._update(InstallProgress("downloading", event.bytes_done, event.bytes_total))

    def _on_state(self, event: DownloadStateChangedEvent) -> None:
        if event.state in ("cancelled", "paused"):
            self._update(InstallProgress("cancelled"))

    def _on_completed(self, _event: DownloadCompletedEvent) -> None:
        _logger.info("Writing model installed in %s.", self._paths.root)
        self._update(InstallProgress("completed"))

    def _on_failed(self, event: DownloadFailedEvent) -> None:
        self._update(InstallProgress("failed", error=event.error))

    def _update(self, progress: InstallProgress) -> None:
        self._progress = progress
        self._on_change()
