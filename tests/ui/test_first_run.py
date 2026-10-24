"""First-run setup is served to the shell; the app waits, pumping Tk, until setup is done."""

import json
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.fakes.config import FakeConfig
from tests.fakes.hardware import FakeHardwareDetector
from tests.fakes.licensing import AllowAllLicensePolicy
from tests.fakes.model_manager import FakeModelManager
from skryb.__main__ import SETUP_NOT_FINISHED, _run_first_time_setup
from skryb.core.event_bus import SimpleEventBus
from skryb.transcription.engine_factory import build_engine_registry
from skryb.ui.api_server import ApiServer
from skryb.ui.event_hub import EventHub
from skryb.ui.launcher import SHELL_NOT_BUILT, SHELL_WOULD_NOT_START, ShellLauncher

_PARAKEET = "parakeet-tdt-0.6b-v3"
# Generous: the point is that setup returns at all, not how fast.
_RETURNS_WITHIN_SECONDS = 10.0


class _Shell:
    def __init__(self) -> None:
        self.exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code


@pytest.fixture()
def server() -> Iterator[ApiServer]:
    served = ApiServer(EventHub(SimpleEventBus()), poll_seconds=0.02)
    served.start()
    yield served
    served.stop()


def _post(server: ApiServer, path: str) -> int:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}{path}", method="POST",
        headers={"Authorization": f"Bearer {server.token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status


def _app(server: ApiServer) -> dict[str, object]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.port}/api/app",
        headers={"Authorization": f"Bearer {server.token}"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _exe(tmp_path: Path) -> Path:
    exe = tmp_path / "skryb-ui.exe"
    exe.write_bytes(b"MZ")
    return exe


def _setup(
    tk_root, tmp_path: Path, server: ApiServer, launcher: ShellLauncher, config: FakeConfig,
) -> float:
    started = time.monotonic()
    _run_first_time_setup(
        config,
        FakeModelManager(installed={_PARAKEET: True}),
        SimpleEventBus(),
        tk_root,
        build_engine_registry(),
        FakeHardwareDetector(),
        tmp_path / "dictionary.json",
        AllowAllLicensePolicy(),
        api_server=server,
        launcher=launcher,
        diagnostics_dir=tmp_path,
    )
    return time.monotonic() - started


def _after(seconds: float, action: Callable[[], None]) -> threading.Thread:
    def run() -> None:
        time.sleep(seconds)
        action()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_setup_is_served_to_the_shell_and_waits_until_the_shell_says_it_is_done(
    tk_root, tmp_path: Path, server: ApiServer,
) -> None:
    shell = _Shell()
    commands: list[list[str]] = []
    seen: list[dict[str, object]] = []

    def spawn(args: list[str]) -> _Shell:
        commands.append(args)
        _after(0.3, lambda: (seen.append(_app(server)), _post(server, "/api/setup/done")))
        return shell

    launcher = ShellLauncher(
        server, lambda message: pytest.fail(message),
        spawn=spawn, candidates=lambda: [_exe(tmp_path)],
    )
    config = FakeConfig({"transcription.model_size": _PARAKEET})

    took = _setup(tk_root, tmp_path, server, launcher, config)

    assert 0.3 <= took < _RETURNS_WITHIN_SECONDS
    assert commands[0][-2:] == ["--page", "setup"]
    assert seen[0]["first_run"] is True
    assert config.save_count >= 1
    assert shell.exit_code is None


def test_closing_the_shell_before_a_model_is_installed_ends_the_wait_and_says_the_app_will_close(
    tk_root, tmp_path: Path, server: ApiServer, monkeypatch,
) -> None:
    shown: list[str] = []
    monkeypatch.setattr("skryb.__main__._show_setup_problem", shown.append)
    shell = _Shell()
    launcher = ShellLauncher(
        server, lambda message: pytest.fail(message),
        spawn=lambda _args: shell, candidates=lambda: [_exe(tmp_path)],
    )
    _after(0.3, lambda: setattr(shell, "exit_code", 0))
    not_downloaded = FakeConfig({"transcription.model_size": "crisperwhisper-2.0-turbo"})

    took = _setup(tk_root, tmp_path, server, launcher, not_downloaded)

    assert 0.3 <= took < _RETURNS_WITHIN_SECONDS
    assert shown == [SETUP_NOT_FINISHED]


@pytest.mark.parametrize(
    ("built", "message"), [(True, SHELL_WOULD_NOT_START), (False, SHELL_NOT_BUILT)],
    ids=["will-not-start", "not-built"],
)
def test_a_shell_that_never_starts_is_reported_and_setup_does_not_hang(
    tk_root, tmp_path: Path, server: ApiServer, built: bool, message: str,
) -> None:
    notices: list[str] = []

    def refuse(_args: list[str]) -> _Shell:
        raise FileNotFoundError(2, "The system cannot find the file specified")

    launcher = ShellLauncher(
        server, notices.append, spawn=refuse,
        candidates=lambda: [_exe(tmp_path) if built else tmp_path / "missing.exe"],
    )

    took = _setup(tk_root, tmp_path, server, launcher, FakeConfig())

    assert took < 2.0
    assert notices == [message]
