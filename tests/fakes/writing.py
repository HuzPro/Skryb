import json
import subprocess
import threading
from collections.abc import Callable
from typing import IO

from skryb.downloads.chunked_file_downloader import FileDownloadPlan
from skryb.writing.client import WritingUnavailable


class UnreachableDownloader:
    """Fails the way requests does with no network, URL and all."""

    def download(
        self,
        plan: FileDownloadPlan,
        on_progress: Callable[[int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        raise ConnectionError(
            "HTTPSConnectionPool(host='example.invalid', port=443): Max retries exceeded "
            f"with url: {plan.url}\n(Caused by NameResolutionError(getaddrinfo failed))",
        )


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class ScriptedClient:
    """Answers like the writing model would, and records what it was asked.

    With no answer it echoes the dictation back, as a model that found
    nothing to fix would. ``takes`` advances the fake clock, so a slow model
    can be simulated without sleeping.
    """

    def __init__(
        self,
        answer: str | None = None,
        error: WritingUnavailable | None = None,
        clock: FakeClock | None = None,
        takes: float = 0.0,
    ) -> None:
        self.answer = answer
        self.error = error
        self.clock = clock
        self.takes = takes
        self.requests: list[tuple[str, dict[str, object], int, float]] = []

    def complete(self, system: str, user: str, max_tokens: int, timeout: float) -> str:
        payload = json.loads(user)
        self.requests.append((system, payload, max_tokens, timeout))
        if self.clock is not None:
            self.clock.now += self.takes
        if self.error is not None:
            raise self.error
        if self.answer is not None:
            return self.answer
        return str(payload.get("dictation", payload.get("selected_text", "")))


BLANK = "   \n\t "
UNICODE = "Tell Aïsha the café in 東京 opens at 9, and say مرحبا to José."
MULTILINE = "First line for Jon.\nSecond line, um, with a list.\n\nThird paragraph here."
THREE_THOUSAND_WORDS = " ".join(
    f"word{index % 97} and Ayesha said café {index}" for index in range(500)
)
INPUTS = [BLANK, UNICODE, THREE_THOUSAND_WORDS, MULTILINE]
INPUT_IDS = ["blank", "unicode", "three-thousand-words", "multi-line"]


class FakeChild:
    """A server process: loads, maybe crashes, dies when killed."""

    def __init__(self, pid: int, exits_while_loading: int | None = None) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.killed = False
        self._exited = threading.Event()
        if exits_while_loading is not None:
            self.exit(exits_while_loading)

    def exit(self, code: int) -> None:
        self.returncode = code
        self._exited.set()

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if not self._exited.wait(timeout):
            raise subprocess.TimeoutExpired("llama-server", timeout or 0.0)
        assert self.returncode is not None
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.exit(-9)


class FakeLaunch:
    """Hands out scripted children and writes what a crashing server would log."""

    def __init__(self, children: list[FakeChild], log_line: str = "") -> None:
        self.children = list(children)
        self.launched: list[FakeChild] = []
        self.args: list[list[str]] = []
        self.log_line = log_line

    def __call__(
        self,
        args: list[str],
        *,
        stdin: int,
        stdout: IO[str],
        stderr: int,
        creationflags: int,
        cwd: str,
    ) -> FakeChild:
        child = self.children.pop(0)
        self.args.append(args)
        self.launched.append(child)
        if self.log_line:
            stdout.write(f"{self.log_line} (pid {child.pid})\n")
        return child
