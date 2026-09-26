"""Runs ``llama-server.exe`` as a supervised child process.

The child gets a free loopback port, a random API key, no console window and
a log file in the diagnostics session, so the reason for any exit is on disk.
A crash is restarted after 2, 10 and 60 seconds; a fourth exit gives up and
reports the end of the log.
"""

import logging
import secrets
import socket
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import IO, Protocol

from voicetotext.writing.client import Endpoint, loopback_opener
from voicetotext.writing.win32_job import bind_to_app_lifetime

_logger = logging.getLogger(__name__)

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_BASE_FLAGS = (
    "-c", "8192", "-ngl", "99", "-np", "1", "--no-webui", "--offline",
)
# Only passed when the build's --help lists them, so an older custom build
# still starts. The speculative settings were picked by measurement: n-gram
# lookup drafts from the dictation itself, which cleanup mostly copies.
_OPTIONAL_FLAGS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("--reasoning [on|off",), ("--reasoning", "off")),
    (("--flash-attn [on|off",), ("-fa", "on")),
    (
        ("ngram-simple", "--spec-ngram-simple-size-n", "--spec-ngram-simple-size-m"),
        (
            "--spec-type", "ngram-simple",
            "--spec-ngram-simple-size-n", "3",
            "--spec-ngram-simple-size-m", "24",
        ),
    ),
)
BACKOFF_SECONDS = (2.0, 10.0, 60.0)
READINESS_TIMEOUT_SECONDS = 60.0
_HEALTH_POLL_SECONDS = 0.2
_HEALTH_TIMEOUT_SECONDS = 1.0
_HELP_TIMEOUT_SECONDS = 30.0
_KILL_WAIT_SECONDS = 5.0
_LOG_TAIL_LINES = 20
_LOG_TAIL_BYTES = 16_384


class ServerState(Enum):
    NOT_INSTALLED = "not_installed"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass(frozen=True)
class ServerStatus:
    state: ServerState
    detail: str = ""


@dataclass(frozen=True)
class ServerFiles:
    runtime: Path
    model: Path


class ChildProcess(Protocol):
    @property
    def pid(self) -> int: ...
    @property
    def returncode(self) -> int | None: ...
    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def kill(self) -> None: ...


class Launch(Protocol):
    def __call__(
        self,
        args: list[str],
        *,
        stdin: int,
        stdout: IO[str],
        stderr: int,
        creationflags: int,
        cwd: str,
    ) -> ChildProcess: ...


@dataclass
class _Run:
    """One supervised lifetime: started by ``start``, ended by ``stop``."""

    stop: threading.Event = field(default_factory=threading.Event)
    process: ChildProcess | None = None


def _launch_detached(
    args: list[str],
    *,
    stdin: int,
    stdout: IO[str],
    stderr: int,
    creationflags: int,
    cwd: str,
) -> ChildProcess:
    return subprocess.Popen(
        args, stdin=stdin, stdout=stdout, stderr=stderr,
        creationflags=creationflags, cwd=cwd,
    )


def _is_healthy(endpoint: Endpoint) -> bool:
    request = urllib.request.Request(
        endpoint.base_url + "/health",
        headers={"Authorization": f"Bearer {endpoint.api_key}"},
    )
    try:
        with loopback_opener().open(request, timeout=_HEALTH_TIMEOUT_SECONDS) as response:
            return bool(response.status == 200)
    except OSError:
        return False


def help_text(runtime: Path) -> str:
    """What the build says about its own flags, or ``""``."""
    try:
        finished = subprocess.run(
            [str(runtime), "--help"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_HELP_TIMEOUT_SECONDS,
            creationflags=_CREATE_NO_WINDOW,
            cwd=str(runtime.parent),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _logger.warning("Could not read the writing server's flags: %s", exc)
        return ""
    return finished.stdout + finished.stderr


def launch_flags(help_output: str) -> tuple[str, ...]:
    optional = tuple(
        flag
        for markers, flags in _OPTIONAL_FLAGS
        if all(marker in help_output for marker in markers)
        for flag in flags
    )
    return _BASE_FLAGS + optional


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def log_tail(path: Path, lines: int = _LOG_TAIL_LINES) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - _LOG_TAIL_BYTES))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    kept = [line for line in text.splitlines() if line.strip()]
    return "\n".join(kept[-lines:])


def last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _kill(process: ChildProcess | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.kill()
        process.wait(timeout=_KILL_WAIT_SECONDS)
    except (OSError, subprocess.SubprocessError) as exc:
        _logger.warning("Could not stop the writing server: %s", exc)


def _ignore_status(_status: ServerStatus) -> None:
    return None


class LlamaServer:
    """Keeps one writing-model server alive while the app runs."""

    def __init__(
        self,
        locate: Callable[[], ServerFiles | None],
        log_path: Path,
        warm_up: Callable[[Endpoint], None],
        *,
        launch: Launch = _launch_detached,
        is_healthy: Callable[[Endpoint], bool] = _is_healthy,
        read_help: Callable[[Path], str] = help_text,
        bind: Callable[[int], None] = bind_to_app_lifetime,
        backoff: tuple[float, ...] = BACKOFF_SECONDS,
        readiness_timeout: float = READINESS_TIMEOUT_SECONDS,
        poll_seconds: float = _HEALTH_POLL_SECONDS,
    ) -> None:
        self._locate = locate
        self._log_path = log_path
        self._warm_up = warm_up
        self._launch = launch
        self._is_healthy = is_healthy
        self._read_help = read_help
        self._bind = bind
        self._backoff = backoff
        self._readiness_timeout = readiness_timeout
        self._poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._run: _Run | None = None
        self._endpoint: Endpoint | None = None
        self._status = ServerStatus(ServerState.STOPPED)
        self._listeners: list[Callable[[ServerStatus], None]] = []
        self._flags_by_runtime: dict[Path, tuple[str, ...]] = {}
        self.ready_event = threading.Event()

    @property
    def status(self) -> ServerStatus:
        return self._status

    @property
    def log_path(self) -> Path:
        return self._log_path

    def endpoint(self) -> Endpoint | None:
        return self._endpoint

    def subscribe(self, listener: Callable[[ServerStatus], None]) -> None:
        self._listeners.append(listener)

    def start(self) -> None:
        """Start supervising on a daemon thread; a no-op while already running."""
        with self._lock:
            if self._run is not None and not self._run.stop.is_set():
                return
            run = _Run()
            self._run = run
        self._publish(run, ServerStatus(ServerState.STARTING))
        threading.Thread(
            target=self._supervise, args=(run,), name="writing-server", daemon=True,
        ).start()

    def stop(self) -> None:
        with self._lock:
            run, self._run = self._run, None
            self._endpoint = None
            self.ready_event.clear()
        if run is None:
            return
        run.stop.set()
        _kill(run.process)
        self._set_status(ServerStatus(ServerState.STOPPED))

    def restart(self) -> None:
        self.stop()
        self.start()

    def dispose(self) -> None:
        self.stop()

    def _supervise(self, run: _Run) -> None:
        try:
            self._keep_alive(run)
        except Exception as exc:  # noqa: BLE001 - a dead supervisor must say why
            _logger.exception("The writing server supervisor failed.")
            self._publish(run, ServerStatus(ServerState.FAILED, str(exc)))
        finally:
            run.stop.set()

    def _keep_alive(self, run: _Run) -> None:
        for delay in (*self._backoff, None):
            files = self._locate()
            if files is None:
                self._publish(run, ServerStatus(ServerState.NOT_INSTALLED))
                return
            ended = self._serve_until_exit(run, files)
            if run.stop.is_set():
                return
            tail = log_tail(self._log_path)
            _logger.warning(
                "The writing server stopped: %s. Last lines of its log:\n%s", ended, tail,
            )
            if delay is None or ended.startswith("it did not become ready"):
                self._publish(run, ServerStatus(ServerState.FAILED, tail or ended))
                return
            self._publish(
                run,
                ServerStatus(ServerState.STARTING, f"Restarting in {delay:g} s: {ended}"),
            )
            if run.stop.wait(delay):
                return

    def _serve_until_exit(self, run: _Run, files: ServerFiles) -> str:
        """Run the child until it exits; say how it ended."""
        endpoint = Endpoint(f"http://127.0.0.1:{_free_port()}", secrets.token_urlsafe(32))
        process = self._start_child(files, endpoint)
        run.process = process
        if run.stop.is_set():
            _kill(process)
            return "stopped"
        self._bind_quietly(process)
        if not self._wait_until_healthy(run, process, endpoint):
            if process.poll() is None:
                _kill(process)
                return f"it did not become ready within {self._readiness_timeout:g} s"
            return f"it exited with code {process.returncode} while loading"
        self._warm_up_quietly(endpoint)
        self._publish(run, ServerStatus(ServerState.READY), endpoint)
        code = process.wait()
        self._publish(run, ServerStatus(ServerState.STARTING, "The server exited."))
        return f"it exited with code {code}"

    def _start_child(self, files: ServerFiles, endpoint: Endpoint) -> ChildProcess:
        port = endpoint.base_url.rsplit(":", 1)[1]
        flags = self._flags_for(files.runtime)
        args = [
            str(files.runtime), "-m", str(files.model),
            "--host", "127.0.0.1", "--port", port, *flags,
        ]
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n=== {datetime.now(UTC).isoformat()} starting: {' '.join(args)}\n")
            log.flush()
            return self._launch(
                [*args, "--api-key", endpoint.api_key],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=_CREATE_NO_WINDOW,
                cwd=str(files.runtime.parent),
            )

    def _flags_for(self, runtime: Path) -> tuple[str, ...]:
        flags = self._flags_by_runtime.get(runtime)
        if flags is None:
            flags = launch_flags(self._read_help(runtime))
            self._flags_by_runtime[runtime] = flags
        return flags

    def _bind_quietly(self, process: ChildProcess) -> None:
        try:
            self._bind(process.pid)
        except OSError as exc:
            _logger.warning("The writing server is not tied to the app's lifetime: %s", exc)

    def _wait_until_healthy(
        self, run: _Run, process: ChildProcess, endpoint: Endpoint,
    ) -> bool:
        deadline = time.monotonic() + self._readiness_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            if self._is_healthy(endpoint):
                return True
            if run.stop.wait(self._poll_seconds):
                return False
        return False

    def _warm_up_quietly(self, endpoint: Endpoint) -> None:
        started = time.monotonic()
        try:
            self._warm_up(endpoint)
        except Exception as exc:  # noqa: BLE001 - a cold first dictation is fine
            _logger.warning("The writing model warm-up failed: %s", exc)
            return
        _logger.info("Writing model ready; warm-up took %.2fs.", time.monotonic() - started)

    def _publish(
        self, run: _Run, status: ServerStatus, endpoint: Endpoint | None = None,
    ) -> None:
        with self._lock:
            if run is not self._run:
                return
            self._endpoint = endpoint
            if endpoint is None:
                self.ready_event.clear()
            else:
                self.ready_event.set()
        self._set_status(status)

    def _set_status(self, status: ServerStatus) -> None:
        self._status = status
        _logger.info("Writing server: %s. %s", status.state.value, last_line(status.detail))
        for listener in list(self._listeners):
            listener(status)
