from unittest.mock import MagicMock

import pytest

from voicetotext.downloads.requests_http_client import RequestsHttpClient


class _FakeResponse:
    def __init__(self, status_code=200, headers=None, content=b""):
        self.status_code = status_code
        self.headers = headers or {}
        self._content = content

    @property
    def content(self):
        return self._content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class _FakeSession:
    def __init__(self):
        self.head_calls = []
        self.get_calls = []
        self.head_response = _FakeResponse(
            status_code=200,
            headers={
                "Content-Length": "1000",
                "Accept-Ranges": "bytes",
                "ETag": '"abc123"',
            },
        )
        self.get_response = _FakeResponse(status_code=206, content=b"RANGE-DATA")

    def head(self, url, allow_redirects=True, timeout=None):
        self.head_calls.append((url, allow_redirects))
        return self.head_response

    def get(self, url, headers=None, stream=False, timeout=None):
        self.get_calls.append((url, dict(headers or {}), stream))
        return self.get_response


class TestRequestsHttpClientHead:
    def test_should_return_content_length_etag_and_accept_ranges(self):
        session = _FakeSession()
        client = RequestsHttpClient(session=session)

        result = client.head("https://example.com/file")

        assert result.content_length == 1000
        assert result.etag == '"abc123"'
        assert result.accept_ranges == "bytes"

    def test_should_follow_redirects_on_head(self):
        session = _FakeSession()
        client = RequestsHttpClient(session=session)

        client.head("https://example.com/file")

        assert session.head_calls[0][1] is True

    def test_should_return_none_etag_when_absent(self):
        session = _FakeSession()
        session.head_response = _FakeResponse(
            status_code=200,
            headers={"Content-Length": "100", "Accept-Ranges": "bytes"},
        )
        client = RequestsHttpClient(session=session)

        result = client.head("u")

        assert result.etag is None


class TestRequestsHttpClientGetFull:
    def test_should_return_body_with_no_range_header(self):
        session = _FakeSession()
        session.get_response = _FakeResponse(status_code=200, content=b"FULL-BODY")
        client = RequestsHttpClient(session=session)

        data = client.get_full("u")

        assert data == b"FULL-BODY"
        assert "Range" not in session.get_calls[0][1]
