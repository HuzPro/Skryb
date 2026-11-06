"""Runs llama.cpp's ``llama-server`` for translation: started by the first request, ended with the app.

Restored from the writing model's runner (removed in 0eeb12c). The child gets a free loopback port,
an API key passed in its environment, no console window and a log file, so the reason for any exit
is on disk, and a Win32 job object ends it with the app. A child that has died is started again by
the next request.
"""

import http.client
import json
import logging
import os
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Protocol, cast

from skryb.translation.win32_job import bind_to_app_lifetime

_logger = logging.getLogger(__name__)

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# One slot of 4096 tokens holds the longest gloss call (48 words) with its prompt.
BASE_FLAGS = ("-c", "4096", "-np", "1", "--no-webui", "--offline", "--jinja")
READINESS_TIMEOUT_SECONDS = 180.0
REQUEST_TIMEOUT_SECONDS = 180.0
_LIST_TIMEOUT_SECONDS = 60.0
_HEALTH_POLL_SECONDS = 0.2
_HEALTH_TIMEOUT_SECONDS = 1.0
_KILL_WAIT_SECONDS = 5.0
_LOG_TAIL_BYTES = 16_384
_MAX_ANSWER_BYTES = 4_000_000
_DEVICE_LINE = re.compile(r"^\s*(\w+?\d+):\s*(.+?)\s*\((\d+) MiB", re.MULTILINE)


class ServerError(Exception):
    """The server could not start or answer; ``str()`` is a sentence to show."""


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str


@dataclass(frozen=True)
class ServerLaunch:
    runtime: Path
    model: Path
    flags: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Device:
    """Where the model runs: ``kind`` is GPU, NPU or CPU, ``name`` what llama.cpp calls that device."""

    kind: str
    name: str = ""


class ChildProcess(Protocol):
    @property
    def pid(self) -> int: ...
    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


class StartChild(Protocol):
    def __call__(self, args: list[str], *, env: dict[str, str], stdout: IO[str], cwd: str) -> ChildProcess: ...


def _popen(args: list[str], *, env: dict[str, str], stdout: IO[str], cwd: str) -> ChildProcess:
    return subprocess.Popen(
        args, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=subprocess.STDOUT,
        creationflags=_CREATE_NO_WINDOW, cwd=cwd,
    )


def loopback_opener() -> urllib.request.OpenerDirector:
    """An opener that never routes loopback traffic through a system proxy."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _is_healthy(endpoint: Endpoint) -> bool:
    request = urllib.request.Request(endpoint.base_url + "/health", headers={"Authorization": f"Bearer {endpoint.api_key}"})
    try:
        return _answers_ok(request)
    except OSError:
        return False


def _answers_ok(request: urllib.request.Request) -> bool:
    with loopback_opener().open(request, timeout=_HEALTH_TIMEOUT_SECONDS) as response:
        return bool(response.status == 200)


def list_devices(runtime: Path, env: Mapping[str, str]) -> list[tuple[str, str, int]]:
    """``(id, name, MiB)`` for each device the build can offload to, from ``--list-devices``."""
    try:
        finished = _run_listing(runtime, env)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ServerError(f"llama.cpp could not list its devices: {exc}") from exc
    return [(key, name, int(mib)) for key, name, mib in _DEVICE_LINE.findall(finished.stdout + finished.stderr)]


def _run_listing(runtime: Path, env: Mapping[str, str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [str(runtime), "--list-devices"], env={**os.environ, **env}, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=_LIST_TIMEOUT_SECONDS,
        creationflags=_CREATE_NO_WINDOW, cwd=str(runtime.parent),
    )


def log_tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - _LOG_TAIL_BYTES))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _kill(process: ChildProcess) -> None:
    if process.poll() is not None:
        return
    try:
        _kill_and_wait(process)
    except (OSError, subprocess.SubprocessError) as exc:
        _logger.warning("Could not stop the translation server: %s", exc)


def _kill_and_wait(process: ChildProcess) -> None:
    process.kill()
    process.wait(timeout=_KILL_WAIT_SECONDS)


class LlamaServer:
    """One ``llama-server`` child, started by the first ``endpoint()`` call."""

    def __init__(
        self,
        launch: Callable[[], ServerLaunch],
        log_path: Path,
        *,
        start_child: StartChild = _popen,
        is_healthy: Callable[[Endpoint], bool] = _is_healthy,
        bind: Callable[[int], None] = bind_to_app_lifetime,
        readiness_timeout: float = READINESS_TIMEOUT_SECONDS,
        poll_seconds: float = _HEALTH_POLL_SECONDS,
    ) -> None:
        self._launch = launch
        self.log_path = log_path
        self._start_child = start_child
        self._is_healthy = is_healthy
        self._bind = bind
        self._readiness_timeout = readiness_timeout
        self._poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._child: ChildProcess | None = None
        self._endpoint: Endpoint | None = None

    def endpoint(self) -> Endpoint:
        """The running server's address, starting it first if it is not running."""
        with self._lock:
            if self._child is not None and self._endpoint is not None and self._child.poll() is None:
                return self._endpoint
            self._child, self._endpoint = None, None
            launch = self._launch()
            endpoint = Endpoint(f"http://127.0.0.1:{_free_port()}", secrets.token_urlsafe(32))
            child = self._spawn(launch, endpoint)
            self._bind_quietly(child)
            self._wait_until_healthy(child, endpoint)
            self._child, self._endpoint = child, endpoint
            return endpoint

    def stop(self) -> None:
        with self._lock:
            child, self._child, self._endpoint = self._child, None, None
        if child is not None:
            _kill(child)

    def _spawn(self, launch: ServerLaunch, endpoint: Endpoint) -> ChildProcess:
        port = endpoint.base_url.rsplit(":", 1)[1]
        args = [str(launch.runtime), "-m", str(launch.model), "--host", "127.0.0.1", "--port", port, *BASE_FLAGS, *launch.flags]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== {datetime.now(UTC).isoformat()} starting: {' '.join(args)}\n")
            log.flush()
            return self._start_child(
                args, env={**os.environ, **launch.env, "LLAMA_API_KEY": endpoint.api_key}, stdout=log, cwd=str(launch.runtime.parent),
            )

    def _bind_quietly(self, child: ChildProcess) -> None:
        try:
            self._bind(child.pid)
        except OSError as exc:
            _logger.warning("The translation server is not tied to the app's lifetime: %s", exc)

    def _wait_until_healthy(self, child: ChildProcess, endpoint: Endpoint) -> None:
        deadline = time.monotonic() + self._readiness_timeout
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise ServerError(f"The translation model stopped while loading. {_last_line(log_tail(self.log_path))}")
            if self._is_healthy(endpoint):
                return
            time.sleep(self._poll_seconds)
        _kill(child)
        raise ServerError(f"The translation model did not load within {self._readiness_timeout:g} seconds.")


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def chat(
    endpoint: Endpoint, prompt: str, max_tokens: int, grammar: str | None,
    timeout: float = REQUEST_TIMEOUT_SECONDS, options: Mapping[str, str] | None = None,
) -> str:
    """One greedy chat completion, constrained by *grammar* when one is given; *options* go to the chat template."""
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens, "cache_prompt": True,
    }
    if grammar is not None:
        payload["grammar"] = grammar
    if options:
        payload["chat_template_kwargs"] = dict(options)
    request = urllib.request.Request(
        endpoint.base_url + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {endpoint.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        body = _read_answer(request, timeout)
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise ServerError(f"The translation model did not answer: {exc}") from exc
    return _content_of(body, locked=grammar is not None)


def _read_answer(request: urllib.request.Request, timeout: float) -> bytes:
    with loopback_opener().open(request, timeout=timeout) as response:
        body = cast(bytes, response.read(_MAX_ANSWER_BYTES + 1))
    if len(body) > _MAX_ANSWER_BYTES:
        raise ValueError("the answer was too large")
    return body


def _content_of(body: bytes, *, locked: bool) -> str:
    try:
        finish_reason, content, reasoning = _first_choice(body)
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ServerError("The translation model's answer could not be read.") from exc
    # Jais-3's template opens a thinking block before every answer. A grammar-locked answer can't
    # close it first, so llama.cpp files the whole answer as reasoning.
    if locked and not content:
        content = reasoning
    if finish_reason != "stop" or not isinstance(content, str):
        raise ServerError(f"The translation model's answer was cut short ({finish_reason}).")
    return content


def _first_choice(body: bytes) -> tuple[object, object, object]:
    choice = json.loads(body)["choices"][0]
    message = choice["message"]
    return choice["finish_reason"], message["content"], message.get("reasoning_content")
