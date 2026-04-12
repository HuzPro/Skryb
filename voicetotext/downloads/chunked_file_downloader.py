import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Protocol

from voicetotext.downloads.chunk_plan import plan_chunks
from voicetotext.downloads.sidecar import (
    Sidecar,
    SidecarCorruptError,
    SidecarVersionError,
)
from voicetotext.downloads.state import ChunkState


_DEFAULT_N_CHUNKS = 8


class FileDownloadError(Exception):
    """Raised when a file download cannot be completed (paused, cancelled, or failed)."""


@dataclass(frozen=True)
class FileDownloadPlan:
    url: str
    target: Path
    expected_size: int


@dataclass(frozen=True)
class HeadResult:
    content_length: int
    etag: str | None
    accept_ranges: str | None


class HttpClient(Protocol):
    def head(self, url: str) -> HeadResult: ...
    def stream_range(self, url: str, start: int, end: int) -> Iterable[bytes]: ...
    def get_full(self, url: str) -> bytes: ...


class ChunkedFileDownloader:
    """Orchestrates the download of a single file.

    Probes the server with HEAD, chooses a strategy (multi-chunk vs full GET),
    resumes from a sidecar when available, and writes into ``<target>.part``
    before renaming to *target* on success.
    """

    def __init__(self, http: HttpClient, n_chunks: int = _DEFAULT_N_CHUNKS) -> None:
        self._http = http
        self._n_chunks = n_chunks

    def download(
        self,
        plan: FileDownloadPlan,
        on_progress: Callable[[int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        part_path = plan.target.with_suffix(plan.target.suffix + ".part")
        meta_path = plan.target.with_suffix(plan.target.suffix + ".part.meta")
        plan.target.parent.mkdir(parents=True, exist_ok=True)

        head = self._http.head(plan.url)
        sidecar = self._load_or_create_sidecar(
            meta_path=meta_path,
            part_path=part_path,
            plan=plan,
            head=head,
        )

        supports_ranges = (head.accept_ranges or "").lower() == "bytes"
        if supports_ranges and len(sidecar.chunks) > 1:
            final_sidecar = self._download_multi_chunk(
                plan=plan,
                part_path=part_path,
                meta_path=meta_path,
                sidecar=sidecar,
                on_progress=on_progress,
                cancel_event=cancel_event,
                pause_event=pause_event,
            )
            if not final_sidecar.is_complete:
                raise FileDownloadError(
                    f"Incomplete after all workers returned: "
                    f"{final_sidecar.bytes_done}/{final_sidecar.content_length}"
                )
        else:
            self._download_single_connection(
                plan=plan,
                part_path=part_path,
                meta_path=meta_path,
                on_progress=on_progress,
            )

        self._finalize(plan=plan, part_path=part_path, meta_path=meta_path)

    # ------------------------------------------------------------------
    # Sidecar handling
    # ------------------------------------------------------------------

    def _load_or_create_sidecar(
        self,
        meta_path: Path,
        part_path: Path,
        plan: FileDownloadPlan,
        head: HeadResult,
    ) -> Sidecar:
        if meta_path.exists():
            try:
                existing = Sidecar.load(meta_path)
            except (SidecarCorruptError, SidecarVersionError):
                existing = None
            else:
                if self._sidecar_matches_server(existing, head):
                    return existing
            # Mismatch → wipe and start fresh
            self._wipe_partial(part_path=part_path, meta_path=meta_path)

        return self._create_fresh_sidecar(meta_path=meta_path, plan=plan, head=head)

    @staticmethod
    def _sidecar_matches_server(sidecar: Sidecar, head: HeadResult) -> bool:
        if sidecar.content_length != head.content_length:
            return False
        if head.etag is not None and sidecar.etag is not None:
            return sidecar.etag == head.etag
        return True

    def _create_fresh_sidecar(
        self, meta_path: Path, plan: FileDownloadPlan, head: HeadResult
    ) -> Sidecar:
        total = head.content_length
        supports_ranges = (head.accept_ranges or "").lower() == "bytes"
        if supports_ranges and total > 0:
            ranges = plan_chunks(total_size=total, n_chunks=self._n_chunks)
            chunks = [
                ChunkState(index=r.index, start=r.start, end=r.end, bytes_done=0)
                for r in ranges
            ]
        else:
            chunks = [ChunkState(index=0, start=0, end=max(total - 1, 0), bytes_done=0)]
        sidecar = Sidecar(
            url=plan.url,
            etag=head.etag,
            content_length=total,
            chunks=chunks,
        )
        sidecar.save(meta_path)
        return sidecar

    @staticmethod
    def _wipe_partial(part_path: Path, meta_path: Path) -> None:
        if part_path.exists():
            part_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    # ------------------------------------------------------------------
    # Multi-chunk path
    # ------------------------------------------------------------------

    def _download_multi_chunk(
        self,
        plan: FileDownloadPlan,
        part_path: Path,
        meta_path: Path,
        sidecar: Sidecar,
        on_progress: Callable[[int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> Sidecar:
        self._ensure_part_file_sized(part_path=part_path, size=sidecar.content_length)

        state_lock = threading.Lock()
        current_sidecar = sidecar

        def run_chunk(chunk: ChunkState) -> None:
            if chunk.is_complete:
                return
            if cancel_event.is_set() or pause_event.is_set():
                return

            fetched_start = chunk.start + chunk.bytes_done
            fetched_end = chunk.end
            stream = self._http.stream_range(plan.url, fetched_start, fetched_end)

            cursor = fetched_start
            bytes_in_chunk = chunk.bytes_done
            try:
                with open(part_path, "r+b") as fd:
                    for piece in stream:
                        if not piece:
                            continue
                        if cancel_event.is_set():
                            return
                        fd.seek(cursor)
                        fd.write(piece)
                        cursor += len(piece)
                        bytes_in_chunk += len(piece)
                        nonlocal current_sidecar
                        with state_lock:
                            current_sidecar = current_sidecar.with_chunk_progress(
                                index=chunk.index, bytes_done=bytes_in_chunk
                            )
                            current_sidecar.save(meta_path)
                        on_progress(len(piece))
                        if pause_event.is_set():
                            return
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()

        workers = min(len(sidecar.chunks), self._n_chunks)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_chunk, chunk) for chunk in sidecar.chunks]
            wait(futures, return_when=FIRST_EXCEPTION)

        for future in futures:
            if future.exception() is not None:
                raise FileDownloadError(
                    f"Chunk download failed: {future.exception()}"
                )

        if pause_event.is_set():
            raise FileDownloadError("Paused")
        if cancel_event.is_set():
            raise FileDownloadError("Cancelled")

        return current_sidecar

    @staticmethod
    def _ensure_part_file_sized(part_path: Path, size: int) -> None:
        if part_path.exists() and part_path.stat().st_size == size:
            return
        with open(part_path, "ab") as fd:
            fd.truncate(size)

    # ------------------------------------------------------------------
    # Single-connection fallback
    # ------------------------------------------------------------------

    def _download_single_connection(
        self,
        plan: FileDownloadPlan,
        part_path: Path,
        meta_path: Path,
        on_progress: Callable[[int], None],
    ) -> None:
        data = self._http.get_full(plan.url)
        part_path.write_bytes(data)
        on_progress(len(data))
        if meta_path.exists():
            meta_path.unlink()

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(
        self, plan: FileDownloadPlan, part_path: Path, meta_path: Path
    ) -> None:
        actual_size = part_path.stat().st_size
        if actual_size != plan.expected_size:
            raise FileDownloadError(
                f"Downloaded size {actual_size} != expected {plan.expected_size}"
            )
        if plan.target.exists():
            plan.target.unlink()
        part_path.rename(plan.target)
        if meta_path.exists():
            meta_path.unlink()
