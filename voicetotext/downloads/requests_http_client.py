from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol

from voicetotext.downloads.chunked_file_downloader import HeadResult


_REQUEST_TIMEOUT_SECONDS = 60
_STREAM_CHUNK_BYTES = 256 * 1024


class _Response(Protocol):
    status_code: int
    headers: Mapping[str, str]
    content: bytes

    def raise_for_status(self) -> None: ...
    def iter_content(self, chunk_size: int = ...) -> Iterator[bytes]: ...
    def close(self) -> None: ...


class _Session(Protocol):
    def head(self, url: str, allow_redirects: bool, timeout: float) -> _Response: ...
    def get(self, url: str, headers: dict[str, str], stream: bool, timeout: float) -> _Response: ...


class RequestsHttpClient:
    """:class:`HttpClient` implementation backed by a ``requests.Session``.

    Owns a single session so HTTP/1.1 keep-alive is reused across chunks —
    this is what actually makes multi-chunk downloads fast against HF.
    """

    def __init__(self, session: _Session) -> None:
        self._session = session

    def head(self, url: str) -> HeadResult:
        response = self._session.head(
            url, allow_redirects=True, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        content_length = int(response.headers.get("Content-Length", "0"))
        return HeadResult(
            content_length=content_length,
            etag=response.headers.get("ETag"),
            accept_ranges=response.headers.get("Accept-Ranges"),
        )

    def stream_range(self, url: str, start: int, end: int) -> Iterable[bytes]:
        response = self._session.get(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            stream=True,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 206:
            response.close()
            raise RuntimeError(
                f"Expected 206 Partial Content for {url}, got {response.status_code}"
            )
        return _StreamingBody(response, _STREAM_CHUNK_BYTES)

    def get_full(self, url: str) -> bytes:
        response = self._session.get(
            url, headers={}, stream=False, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        return response.content


class _StreamingBody:
    """Closeable iterator adapter around a streaming ``requests`` response."""

    def __init__(self, response: _Response, chunk_bytes: int) -> None:
        self._response = response
        self._chunk_bytes = chunk_bytes

    def __iter__(self) -> Iterator[bytes]:
        yield from self._response.iter_content(chunk_size=self._chunk_bytes)

    def close(self) -> None:
        self._response.close()
