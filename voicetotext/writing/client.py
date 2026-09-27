import http.client
import json
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

# A cleaned dictation is a few kilobytes; anything near this is not one.
_MAX_ANSWER_BYTES = 1_000_000


class WritingUnavailable(Exception):
    """The model gave no usable answer in time; the caller keeps its own text."""


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str


def loopback_opener() -> urllib.request.OpenerDirector:
    """An opener that never routes loopback traffic through a system proxy."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post_completion(
    endpoint: Endpoint,
    system: str,
    user: str,
    max_tokens: int,
    timeout: float,
    opener: urllib.request.OpenerDirector | None = None,
) -> str:
    """One chat completion, or ``WritingUnavailable``.

    The timeout is the whole budget: the server sends nothing until it has
    finished, so the socket wait is the generation time, and giving up closes
    the connection, which is how the server learns to free the slot.
    """
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        endpoint.base_url + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        body = _read_answer(opener or loopback_opener(), request, timeout)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise WritingUnavailable(f"no answer ({exc})") from exc
    if time.monotonic() - started > timeout:
        raise WritingUnavailable("the answer came after the deadline")
    return _content_of(body)


def _read_answer(
    opener: urllib.request.OpenerDirector,
    request: urllib.request.Request,
    timeout: float,
) -> bytes:
    with opener.open(request, timeout=timeout) as response:
        body = cast(bytes, response.read(_MAX_ANSWER_BYTES + 1))
    if len(body) > _MAX_ANSWER_BYTES:
        raise ValueError("the answer was too large")
    return body


def _content_of(body: bytes) -> str:
    try:
        document = json.loads(body)
        choice = document["choices"][0]
        finish_reason = choice["finish_reason"]
        content = choice["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise WritingUnavailable("the answer was not a chat completion") from exc
    if finish_reason != "stop" or not isinstance(content, str):
        raise WritingUnavailable(f"the answer was cut short ({finish_reason})")
    return content


class WritingClient:
    """Talks to the local writing model, one request at a time.

    A request that finds another in flight gives up at once rather than
    queueing behind it: the caller has a deadline and its own text to fall
    back to.
    """

    def __init__(self, endpoint: Callable[[], Endpoint | None]) -> None:
        self._endpoint = endpoint
        self._busy = threading.Lock()
        self._opener = loopback_opener()

    def is_ready(self) -> bool:
        return self._endpoint() is not None

    def complete(
        self, system: str, user: str, max_tokens: int, timeout: float,
    ) -> str:
        endpoint = self._endpoint()
        if endpoint is None:
            raise WritingUnavailable("the writing model is not running")
        if timeout <= 0:
            raise WritingUnavailable("no time was left")
        if not self._busy.acquire(blocking=False):
            raise WritingUnavailable("the writing model is busy")
        try:
            return post_completion(
                endpoint, system, user, max_tokens, timeout, self._opener,
            )
        finally:
            self._busy.release()
