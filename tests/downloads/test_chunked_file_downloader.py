import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from voicetotext.downloads.chunked_file_downloader import (
    ChunkedFileDownloader,
    FileDownloadError,
    FileDownloadPlan,
)
from voicetotext.downloads.sidecar import Sidecar
from voicetotext.downloads.state import ChunkState


@dataclass
class _HeadResult:
    content_length: int
    etag: str | None
    accept_ranges: str | None


class _FakeHttpClient:
    """Stub for the HTTP client — we don't exercise requests directly here."""

    def __init__(self, head: _HeadResult, bodies: dict[tuple[int, int], bytes]):
        self.head_response = head
        self._bodies = bodies
        self.range_calls: list[tuple[int, int]] = []
        self.full_calls: list[str] = []

    def head(self, url: str) -> _HeadResult:
        return self.head_response

    def get_range(self, url: str, start: int, end: int) -> bytes:
        self.range_calls.append((start, end))
        return self._bodies[(start, end)]

    def stream_range(self, url: str, start: int, end: int):
        self.range_calls.append((start, end))
        # Yield in two pieces so tests exercise the streaming path
        data = self._bodies[(start, end)]
        mid = max(len(data) // 2, 1)
        yield data[:mid]
        yield data[mid:]

    def get_full(self, url: str) -> bytes:
        self.full_calls.append(url)
        # Assemble full body from ranges if possible, else sum of all body slices.
        # For tests we pass a single (0, total-1) key.
        total = self.head_response.content_length
        return self._bodies[(0, total - 1)]


def _make_plan(tmp_path: Path, size: int = 4000) -> FileDownloadPlan:
    return FileDownloadPlan(
        url="https://example.com/file.bin",
        target=tmp_path / "file.bin",
        expected_size=size,
    )


class TestChunkedFileDownloaderHappyPath:
    def test_should_write_full_file_with_multi_chunk(self, tmp_path):
        size = 4000
        body = bytes(range(256)) * (size // 256) + b"Z" * (size % 256)
        assert len(body) == size
        chunk_size = size // 4
        bodies = {
            (0, chunk_size - 1): body[0:chunk_size],
            (chunk_size, 2 * chunk_size - 1): body[chunk_size : 2 * chunk_size],
            (2 * chunk_size, 3 * chunk_size - 1): body[2 * chunk_size : 3 * chunk_size],
            (3 * chunk_size, size - 1): body[3 * chunk_size : size],
        }
        http = _FakeHttpClient(
            head=_HeadResult(content_length=size, etag='"abc"', accept_ranges="bytes"),
            bodies=bodies,
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4)

        downloader.download(
            plan=_make_plan(tmp_path, size),
            on_progress=lambda n: None,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        target = tmp_path / "file.bin"
        assert target.exists()
        assert target.read_bytes() == body
        assert len(http.range_calls) == 4
        # Sidecar removed after successful completion
        assert not (tmp_path / "file.bin.part").exists()
        assert not (tmp_path / "file.bin.part.meta").exists()

    def test_should_aggregate_progress_across_chunks(self, tmp_path):
        size = 1000
        body = b"A" * size
        bodies = {
            (0, 249): body[0:250],
            (250, 499): body[250:500],
            (500, 749): body[500:750],
            (750, 999): body[750:1000],
        }
        http = _FakeHttpClient(
            head=_HeadResult(content_length=size, etag=None, accept_ranges="bytes"),
            bodies=bodies,
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4)
        received: list[int] = []

        downloader.download(
            plan=_make_plan(tmp_path, size),
            on_progress=received.append,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        assert sum(received) == size


class TestChunkedFileDownloaderSingleConnectionFallback:
    def test_should_fall_back_to_full_get_when_accept_ranges_missing(self, tmp_path):
        size = 1000
        body = b"X" * size
        http = _FakeHttpClient(
            head=_HeadResult(content_length=size, etag=None, accept_ranges=None),
            bodies={(0, size - 1): body},
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4)

        downloader.download(
            plan=_make_plan(tmp_path, size),
            on_progress=lambda n: None,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        assert (tmp_path / "file.bin").read_bytes() == body
        assert http.full_calls == ["https://example.com/file.bin"]
        assert http.range_calls == []


class TestChunkedFileDownloaderResume:
    def test_should_resume_from_sidecar_and_not_refetch_completed_chunks(self, tmp_path):
        size = 1000
        body = b"R" * size
        # Seed the .part file with the first half already downloaded
        part = tmp_path / "file.bin.part"
        part.write_bytes(body[:500] + b"\x00" * 500)
        meta = tmp_path / "file.bin.part.meta"
        Sidecar(
            url="https://example.com/file.bin",
            etag='"same"',
            content_length=size,
            chunks=[
                ChunkState(index=0, start=0, end=249, bytes_done=250),
                ChunkState(index=1, start=250, end=499, bytes_done=250),
                ChunkState(index=2, start=500, end=749, bytes_done=0),
                ChunkState(index=3, start=750, end=999, bytes_done=0),
            ],
        ).save(meta)

        bodies = {
            (500, 749): body[500:750],
            (750, 999): body[750:1000],
        }
        http = _FakeHttpClient(
            head=_HeadResult(content_length=size, etag='"same"', accept_ranges="bytes"),
            bodies=bodies,
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4)

        downloader.download(
            plan=_make_plan(tmp_path, size),
            on_progress=lambda n: None,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        # Only the two incomplete chunks were fetched
        assert set(http.range_calls) == {(500, 749), (750, 999)}
        assert (tmp_path / "file.bin").read_bytes() == body

    def test_should_restart_when_etag_changed(self, tmp_path):
        size = 1000
        part = tmp_path / "file.bin.part"
        part.write_bytes(b"O" * 500 + b"\x00" * 500)
        meta = tmp_path / "file.bin.part.meta"
        Sidecar(
            url="https://example.com/file.bin",
            etag='"old"',
            content_length=size,
            chunks=[
                ChunkState(index=0, start=0, end=499, bytes_done=500),
                ChunkState(index=1, start=500, end=999, bytes_done=0),
            ],
        ).save(meta)

        body = b"N" * size
        bodies = {
            (0, 249): body[0:250],
            (250, 499): body[250:500],
            (500, 749): body[500:750],
            (750, 999): body[750:1000],
        }
        http = _FakeHttpClient(
            head=_HeadResult(content_length=size, etag='"new"', accept_ranges="bytes"),
            bodies=bodies,
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4)

        downloader.download(
            plan=_make_plan(tmp_path, size),
            on_progress=lambda n: None,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        # ETag drift → full refetch of all four chunks
        assert len(http.range_calls) == 4
        assert (tmp_path / "file.bin").read_bytes() == body


class TestChunkedFileDownloaderPause:
    def test_should_leave_sidecar_intact_when_paused(self, tmp_path):
        size = 1000
        body = b"P" * size
        bodies = {
            (0, 249): body[0:250],
            (250, 499): body[250:500],
            (500, 749): body[500:750],
            (750, 999): body[750:1000],
        }
        http = _FakeHttpClient(
            head=_HeadResult(content_length=size, etag=None, accept_ranges="bytes"),
            bodies=bodies,
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4)
        pause = threading.Event()
        pause.set()

        with pytest.raises(FileDownloadError):
            downloader.download(
                plan=_make_plan(tmp_path, size),
                on_progress=lambda n: None,
                cancel_event=threading.Event(),
                pause_event=pause,
            )

        assert (tmp_path / "file.bin.part.meta").exists()
        assert not (tmp_path / "file.bin").exists()


class TestChunkedFileDownloaderSizeMismatch:
    def test_should_raise_when_final_size_does_not_match(self, tmp_path):
        size = 1000
        body = b"M" * (size - 100)  # deliberately short
        bodies = {
            (0, 249): body[0:250],
            (250, 499): body[250:500],
            (500, 749): body[500:750],
            (750, 999): body[750:900],
        }
        http = _FakeHttpClient(
            head=_HeadResult(content_length=size, etag=None, accept_ranges="bytes"),
            bodies=bodies,
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4)

        with pytest.raises(FileDownloadError):
            downloader.download(
                plan=_make_plan(tmp_path, size),
                on_progress=lambda n: None,
                cancel_event=threading.Event(),
                pause_event=threading.Event(),
            )
