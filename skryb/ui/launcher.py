"""Opens the UI shell, a separate process, on the page asked for.

Nothing is spawned until the user asks for a window. A second ``open``
while the shell runs spawns again: the shell's single-instance plugin hands
the new page to the window already open and the new process exits.
"""

import logging
import os
import subprocess
import sys
import tkinter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

_logger = logging.getLogger(__name__)

UI_EXE_ENV = "SKRYB_UI_EXE"
SHELL_NOT_BUILT = "The settings window is not built. Run scripts\\build_ui.ps1."
SHELL_WOULD_NOT_START = "The settings window could not be opened. See the log for details."
_EXE_NAME = "skryb-ui.exe"
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_ASFW_ANY = -1
_SETUP_POLL_MS = 100


class ShellProcess(Protocol):
    def poll(self) -> int | None: ...


class ShellAddress(Protocol):
    @property
    def port(self) -> int: ...

    @property
    def token(self) -> str: ...


Spawn = Callable[[list[str]], ShellProcess]


def shell_candidates(environ: Mapping[str, str] = os.environ) -> list[Path]:
    """Where the shell may be, most specific first."""
    configured = environ.get(UI_EXE_ENV, "").strip()
    return [
        *([Path(configured)] if configured else []),
        _PACKAGE_DIR.parent / "ui" / "src-tauri" / "target" / "release" / _EXE_NAME,
        _PACKAGE_DIR / "ui-bin" / _EXE_NAME,
    ]


def spawn_shell(args: list[str]) -> ShellProcess:
    _let_the_shell_come_to_the_front()
    return subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _let_the_shell_come_to_the_front() -> None:
    """Pass on the right to take the foreground that the tray click gave us.

    Without it Windows only flashes the shell's taskbar button, and the
    window already open never comes forward for a second request.
    """
    if sys.platform != "win32":
        return
    import ctypes

    ctypes.windll.user32.AllowSetForegroundWindow(_ASFW_ANY)


class ShellLauncher:
    def __init__(
        self,
        address: ShellAddress,
        notify: Callable[[str], None],
        *,
        spawn: Spawn = spawn_shell,
        candidates: Callable[[], list[Path]] = shell_candidates,
        updates_dir: Path | None = None,
    ) -> None:
        self._address = address
        self._notify = notify
        self._spawn = spawn
        self._candidates = candidates
        self._updates_dir = updates_dir
        self._child: ShellProcess | None = None

    def open(self, page: str, *, restore: bool = False) -> bool:
        """Show *page*; ``False``, after telling the user why, when it cannot.

        ``restore`` reopens the window the last version handed over on
        updating, where it was and on the page it showed.
        """
        candidates = self._candidates()
        exe = next((path for path in candidates if path.is_file()), None)
        if exe is None:
            _logger.warning(
                "No UI shell found; looked for %s", ", ".join(str(path) for path in candidates),
            )
            self._notify(SHELL_NOT_BUILT)
            return False
        command = [
            str(exe),
            "--port", str(self._address.port),
            "--token", self._address.token,
            "--page", page,
        ]
        # Where the installer reports its progress and the window is handed over.
        if self._updates_dir is not None:
            command += ["--updates-dir", str(self._updates_dir)]
            if restore:
                command.append("--restore")
        try:
            self._child = self._spawn(command)
        except (OSError, ValueError):
            _logger.exception("Could not start the UI shell %s", exe)
            self._notify(SHELL_WOULD_NOT_START)
            return False
        _logger.info("Opened the UI shell on the %s page.", page)
        return True

    def is_running(self) -> bool:
        return self._child is not None and self._child.poll() is None


def wait_for_setup(
    root: tkinter.Misc, finished: Callable[[], bool], poll_ms: int = _SETUP_POLL_MS,
) -> None:
    """Keep Tk's events flowing, as ``wait_window`` did, until *finished* says so.

    No closure holds the variable, so it is freed here on the GUI thread
    rather than by the cycle collector on whichever thread it runs on,
    which Tk refuses.
    """
    tick = tkinter.BooleanVar(master=root)
    while not finished():
        root.after(poll_ms, tick.set, True)
        root.wait_variable(tick)
