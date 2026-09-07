import threading
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Protocol

from voicetotext.downloads.chunked_file_downloader import FileDownloadPlan
from voicetotext.downloads.hf_repo_resolver import ResolvedFile


class ModelDownloadError(Exception):
    """Raised when a model job cannot complete (individual file failures bubble up here)."""


# A step that runs once every file is on disk and the snapshot is
# complete, receiving the snapshot directory. Engines whose runtime needs
# a format the repo does not ship build it here rather than on a
# transcription path.
PostInstallStep = Callable[[Path], None]


def _no_post_install(snapshot: Path) -> None:
    """Default for models usable exactly as downloaded."""


class _FileDownloader(Protocol):
    def download(
        self,
        plan: FileDownloadPlan,
        on_progress: Callable[[int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None: ...


@dataclass
class ModelDownloadJob:
    """Downloads every file for a single model, then finalizes the cache dir.

    The job knows how to translate a list of :class:`ResolvedFile` objects
    into the HuggingFace cache layout (``models--org--repo/snapshots/<sha>/``
    plus ``refs/main`` pointing at the commit sha).

    ``post_install`` runs last, after ``refs/main`` makes the snapshot
    loadable. Running it *after* the completion barrier rather than
    before it keeps a failing step from discarding gigabytes of correctly
    downloaded weights: the snapshot stays installed and the step is
    retried by whoever needs its output.
    """

    job_id: str
    repo_id: str
    commit_sha: str
    files: list[ResolvedFile]
    models_dir: Path
    file_downloader: _FileDownloader
    post_install: PostInstallStep = _no_post_install
    _bytes_done: int = field(default=0, init=False)
    _progress_lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def repo_dir(self) -> Path:
        safe_repo = self.repo_id.replace("/", "--")
        return self.models_dir / f"models--{safe_repo}"

    @property
    def snapshot_dir(self) -> Path:
        return self.repo_dir / "snapshots" / self.commit_sha

    @property
    def refs_main(self) -> Path:
        return self.repo_dir / "refs" / "main"

    def run(
        self,
        on_progress: Callable[[int, int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        self._bytes_done = 0
        total = self.total_bytes
        snapshot = self.snapshot_dir
        snapshot.mkdir(parents=True, exist_ok=True)

        for resolved in self.files:
            if cancel_event.is_set():
                raise ModelDownloadError("Cancelled")
            if pause_event.is_set():
                raise ModelDownloadError("Paused")
            plan = FileDownloadPlan(
                url=resolved.url,
                target=snapshot / resolved.filename,
                expected_size=resolved.size,
            )
            try:
                self.file_downloader.download(
                    plan=plan,
                    on_progress=lambda n: self._on_file_progress(n, total, on_progress),
                    cancel_event=cancel_event,
                    pause_event=pause_event,
                )
            except Exception as exc:
                raise ModelDownloadError(
                    f"Failed to download {resolved.filename}: {exc}"
                ) from exc

        self._write_refs_main()
        self.post_install(snapshot)

    def _on_file_progress(
        self,
        delta: int,
        total: int,
        on_progress: Callable[[int, int], None],
    ) -> None:
        with self._progress_lock:
            self._bytes_done += delta
            done = self._bytes_done
        on_progress(done, total)

    def _write_refs_main(self) -> None:
        self.refs_main.parent.mkdir(parents=True, exist_ok=True)
        self.refs_main.write_text(self.commit_sha)
