from collections.abc import Iterator

from voicetotext.downloads.chunked_file_downloader import HeadResult


class FakeHttpClient:
    """Structurally-typed stub for the :class:`HttpClient` protocol.

    Tests pre-program the HEAD response and the byte ranges that
    ``stream_range`` / ``get_full`` will return.  All calls are
    recorded so tests can assert which ranges were requested.
    """

    def __init__(
        self,
        head: HeadResult,
        bodies: dict[tuple[int, int], bytes],
    ) -> None:
        self.head_response = head
        self._bodies = bodies
        self.range_calls: list[tuple[int, int]] = []
        self.full_calls: list[str] = []

    def head(self, url: str) -> HeadResult:
        return self.head_response

    def get_range(self, url: str, start: int, end: int) -> bytes:
        self.range_calls.append((start, end))
        return self._bodies[(start, end)]

    def stream_range(self, url: str, start: int, end: int) -> Iterator[bytes]:
        self.range_calls.append((start, end))
        data = self._bodies[(start, end)]
        mid = max(len(data) // 2, 1)
        yield data[:mid]
        yield data[mid:]

    def get_full(self, url: str) -> bytes:
        self.full_calls.append(url)
        total = self.head_response.content_length
        return self._bodies[(0, total - 1)]
