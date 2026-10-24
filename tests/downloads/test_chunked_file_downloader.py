import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from skryb.downloads.chunked_file_downloader import (
    ChunkedFileDownloader,
    FileDownloadError,
    FileDownloadPlan,
)
from skryb.downloads.sidecar import Sidecar
from skryb.downloads.state import ChunkState


@dataclass
class _HeadResult:
    content_length: int
    etag: str | None
    accept_ranges: str | None


class _FakeHttpClient:
    """Stub for the HTTP client - we don't exercise requests directly here."""

    def __init__(self, head: _HeadResult, bodies: dict[tuple[int, int], bytes]) -> None:
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
        downloader = ChunkedFileDownloader(http=http, n_chunks=4, retry_delays=())

        with pytest.raises(FileDownloadError):
            downloader.download(
                plan=_make_plan(tmp_path, size),
                on_progress=lambda n: None,
                cancel_event=threading.Event(),
                pause_event=threading.Event(),
            )


class _MidStreamFailingHttpClient:
    """First call to a specific chunk yields one piece, then raises. Then a healthy fallback."""

    def __init__(
        self,
        head: _HeadResult,
        bodies: dict[tuple[int, int], bytes],
        failing_range: tuple[int, int],
    ) -> None:
        self.head_response = head
        self._bodies = bodies
        self._failing_range = failing_range
        self._failed_once = False
        self.range_calls: list[tuple[int, int]] = []

    def head(self, url):
        return self.head_response

    def stream_range(self, url, start, end):
        self.range_calls.append((start, end))
        data = self._bodies[(start, end)]
        mid = max(len(data) // 2, 1)
        if (start, end) == self._failing_range and not self._failed_once:
            self._failed_once = True

            def _broken_generator():
                yield data[:mid]
                raise RuntimeError("connection reset mid-stream")

            return _broken_generator()
        # Healthy stream (used after failure for resume; also used for other chunks)
        return iter([data[:mid], data[mid:]])

    def get_full(self, url):
        raise AssertionError("not used in this test")


class TestChunkedFileDownloaderMidStreamError:
    def test_should_resume_after_stream_raises_mid_piece_without_byte_loss(self, tmp_path):
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
        failing_range = (chunk_size, 2 * chunk_size - 1)
        http = _MidStreamFailingHttpClient(
            head=_HeadResult(content_length=size, etag='"abc"', accept_ranges="bytes"),
            bodies=bodies,
            failing_range=failing_range,
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4, retry_delays=())
        plan = _make_plan(tmp_path, size)

        with pytest.raises(FileDownloadError):
            downloader.download(
                plan=plan,
                on_progress=lambda n: None,
                cancel_event=threading.Event(),
                pause_event=threading.Event(),
            )

        # Second run with extended bodies for resumed tail of the failed chunk
        first_piece_len = max((chunk_size) // 2, 1)
        resumed_start = chunk_size + first_piece_len
        resumed_end = 2 * chunk_size - 1
        bodies[(resumed_start, resumed_end)] = body[resumed_start : resumed_end + 1]

        downloader.download(
            plan=plan,
            on_progress=lambda n: None,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        target = tmp_path / "file.bin"
        assert target.exists()
        assert target.read_bytes() == body
        # Verify the failed chunk only re-requested its missing tail on resume
        assert (resumed_start, resumed_end) in http.range_calls

    def test_a_dropped_connection_resumes_that_chunk_in_the_same_download(self, tmp_path):
        size = 4000
        body = bytes(range(256)) * (size // 256) + b"Z" * (size % 256)
        chunk_size = size // 4
        bodies = {
            (i * chunk_size, (i + 1) * chunk_size - 1): body[i * chunk_size : (i + 1) * chunk_size]
            for i in range(4)
        }
        failing_range = (chunk_size, 2 * chunk_size - 1)
        resumed_start = chunk_size + chunk_size // 2
        bodies[(resumed_start, 2 * chunk_size - 1)] = body[resumed_start : 2 * chunk_size]
        http = _MidStreamFailingHttpClient(
            head=_HeadResult(content_length=size, etag='"abc"', accept_ranges="bytes"),
            bodies=bodies,
            failing_range=failing_range,
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4, retry_delays=(0.0,))
        received: list[int] = []

        downloader.download(
            plan=_make_plan(tmp_path, size),
            on_progress=received.append,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        assert (tmp_path / "file.bin").read_bytes() == body
        assert (resumed_start, 2 * chunk_size - 1) in http.range_calls
        assert sum(received) == size

    def test_a_connection_that_keeps_dropping_without_progress_fails_after_the_retries(self, tmp_path):
        class _AlwaysReset(_FakeHttpClient):
            def stream_range(self, url, start, end):
                self.range_calls.append((start, end))
                raise ConnectionResetError(10054, "An existing connection was forcibly closed")

        http = _AlwaysReset(
            head=_HeadResult(content_length=1000, etag=None, accept_ranges="bytes"), bodies={},
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4, retry_delays=(0.0, 0.0))

        with pytest.raises(FileDownloadError, match="forcibly closed"):
            downloader.download(
                plan=_make_plan(tmp_path, 1000),
                on_progress=lambda n: None,
                cancel_event=threading.Event(),
                pause_event=threading.Event(),
            )

        assert len(http.range_calls) == 4 * 3

    def test_a_chunk_that_keeps_making_progress_is_never_given_up_on(self, tmp_path):
        size = 1000
        body = bytes(i % 251 for i in range(size))

        class _TenBytesAtATime(_FakeHttpClient):
            def stream_range(self, url, start, end):
                self.range_calls.append((start, end))
                yield body[start : min(start + 10, end + 1)]
                if start + 10 <= end:
                    raise ConnectionResetError("dropped")

        http = _TenBytesAtATime(
            head=_HeadResult(content_length=size, etag=None, accept_ranges="bytes"), bodies={},
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4, retry_delays=(0.0,))

        downloader.download(
            plan=_make_plan(tmp_path, size),
            on_progress=lambda n: None,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        assert (tmp_path / "file.bin").read_bytes() == body
        assert len(http.range_calls) == 100

    def test_a_failed_first_request_for_the_file_size_is_retried(self, tmp_path):
        size = 1000
        body = b"H" * size

        class _HeadFailsOnce(_FakeHttpClient):
            heads = 0

            def head(self, url):
                self.heads += 1
                if self.heads == 1:
                    raise ConnectionResetError("dropped")
                return self.head_response

        http = _HeadFailsOnce(
            head=_HeadResult(content_length=size, etag=None, accept_ranges=None),
            bodies={(0, size - 1): body},
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4, retry_delays=(0.0,))

        downloader.download(
            plan=_make_plan(tmp_path, size),
            on_progress=lambda n: None,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
        )

        assert (tmp_path / "file.bin").read_bytes() == body
        assert http.heads == 2

    def test_cancelling_during_a_retry_wait_stops_at_once(self, tmp_path):
        cancel = threading.Event()

        class _ResetThenCancel(_FakeHttpClient):
            def stream_range(self, url, start, end):
                self.range_calls.append((start, end))
                cancel.set()
                raise ConnectionResetError("dropped")

        http = _ResetThenCancel(
            head=_HeadResult(content_length=1000, etag=None, accept_ranges="bytes"), bodies={},
        )
        downloader = ChunkedFileDownloader(http=http, n_chunks=4, retry_delays=(60.0,))

        with pytest.raises(FileDownloadError, match="Cancelled"):
            downloader.download(
                plan=_make_plan(tmp_path, 1000),
                on_progress=lambda n: None,
                cancel_event=cancel,
                pause_event=threading.Event(),
            )

    def test_should_preserve_bytes_when_sidecar_file_is_corrupted_between_runs(self, tmp_path):
        size = 1000
        body = b"R" * size
        # Pretend first chunk already downloaded; .part has correct first 250 bytes
        part = tmp_path / "file.bin.part"
        part.write_bytes(body[:250] + b"\x00" * 750)
        meta = tmp_path / "file.bin.part.meta"

        # First save establishes a baseline; the second save will leave the first as .bak
        Sidecar(
            url="https://example.com/file.bin",
            etag='"same"',
            content_length=size,
            chunks=[
                ChunkState(index=0, start=0, end=249, bytes_done=250),
                ChunkState(index=1, start=250, end=499, bytes_done=0),
                ChunkState(index=2, start=500, end=749, bytes_done=0),
                ChunkState(index=3, start=750, end=999, bytes_done=0),
            ],
        ).save(meta)
        Sidecar(
            url="https://example.com/file.bin",
            etag='"same"',
            content_length=size,
            chunks=[
                ChunkState(index=0, start=0, end=249, bytes_done=250),
                ChunkState(index=1, start=250, end=499, bytes_done=0),
                ChunkState(index=2, start=500, end=749, bytes_done=0),
                ChunkState(index=3, start=750, end=999, bytes_done=0),
            ],
        ).save(meta)
        # Corrupt the primary sidecar file. The .bak written by save() should survive.
        meta.write_text("{broken", encoding="utf-8")

        bodies = {
            (250, 499): body[250:500],
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

        # Recovered via .bak: the already-downloaded first chunk is NOT refetched
        assert (0, 249) not in http.range_calls
        assert (tmp_path / "file.bin").read_bytes() == body

    def test_should_flush_part_file_before_recording_offset_in_sidecar(self, tmp_path):
        size = 1000
        body = b"F" * size
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

        part_path = tmp_path / "file.bin.part"
        chunk_by_index = {
            0: (0, 249),
            1: (250, 499),
            2: (500, 749),
            3: (750, 999),
        }

        from skryb.downloads import chunked_file_downloader as cfd

        original_update = cfd._SharedDownloadState.update_chunk

        def _verifying_update(self, index, bytes_done, meta_path):
            start_offset, _ = chunk_by_index[index]
            full_chunk_body = bodies[chunk_by_index[index]]
            expected = full_chunk_body[:bytes_done]
            # bytes must be on disk BEFORE the sidecar claims them
            with open(part_path, "rb") as fd:
                fd.seek(start_offset)
                actual = fd.read(bytes_done)
            assert actual == expected, (
                f"Sidecar claimed {bytes_done} bytes for chunk {index} "
                f"but disk only had {len(actual.rstrip(bytes([0])))} non-zero bytes"
            )
            return original_update(self, index, bytes_done, meta_path)

        with patch.object(cfd._SharedDownloadState, "update_chunk", _verifying_update):
            downloader.download(
                plan=_make_plan(tmp_path, size),
                on_progress=lambda n: None,
                cancel_event=threading.Event(),
                pause_event=threading.Event(),
            )

        assert (tmp_path / "file.bin").read_bytes() == body
