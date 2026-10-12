"""Which commit of the source checkout this process runs, and whether it has moved.

The app runs straight from a git checkout, so merging to main changes the
code on disk but not the process already running it. The tray menu names
the running commit, and the one a restart would pick up once they differ.
Reads the files under .git directly; there may be no git on PATH.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

_logger = logging.getLogger(__name__)

_SHORT = 7


@dataclass(frozen=True)
class Build:
    commit: str
    # Empty when the checkout is on a detached commit.
    branch: str = ""

    @property
    def label(self) -> str:
        short = self.commit[:_SHORT]
        return f"{self.branch} {short}" if self.branch else short


def source_checkout() -> Path:
    """The folder holding this package's checkout, when it runs from one."""
    return Path(__file__).resolve().parents[2]


def read_build(checkout: Path) -> Build | None:
    """The commit checked out in ``checkout``, or None when it is not a git checkout."""
    try:
        git_dir = _git_dir(checkout)
        if git_dir is None:
            return None
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return Build(commit=head) if head else None
        ref = head.removeprefix("ref:").strip()
        commit = _resolve(ref, git_dir, _common_dir(git_dir))
        if commit is None:
            return None
        return Build(commit=commit, branch=ref.removeprefix("refs/heads/"))
    except (OSError, UnicodeDecodeError):
        _logger.debug("Reading the checkout's commit failed.", exc_info=True)
        return None


class RunningBuild:
    """The commit read at start-up, held against the one on disk now."""

    def __init__(self, checkout: Path) -> None:
        self._checkout = checkout
        self._started = read_build(checkout)

    def status(self) -> str:
        if self._started is None:
            return ""
        on_disk = read_build(self._checkout)
        if on_disk is None or on_disk.commit == self._started.commit:
            return self._started.label
        return f"{self._started.label}, restart for {on_disk.label}"


def source_changed_since(started_at: float, package: Path | None = None) -> bool:
    """Whether any of the app's Python files was written after *started_at*.

    A pull, a branch switch and an edit in place all rewrite the files they
    touch, so this sees each of them.
    """
    root = package or Path(__file__).resolve().parents[1]
    return any(_modified_at(path) > started_at for path in root.rglob("*.py"))


def _modified_at(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        # Deleted since the folder was listed.
        return 0.0


def _git_dir(checkout: Path) -> Path | None:
    dot_git = checkout / ".git"
    if dot_git.is_dir():
        return dot_git
    if dot_git.is_file():
        # A worktree or submodule: ".git" names the real git folder.
        pointer = dot_git.read_text(encoding="utf-8").strip()
        if pointer.startswith("gitdir:"):
            return (checkout / pointer.removeprefix("gitdir:").strip()).resolve()
    return None


def _common_dir(git_dir: Path) -> Path:
    """Where a worktree's branches live: the main checkout's git folder."""
    pointer = git_dir / "commondir"
    if pointer.is_file():
        return (git_dir / pointer.read_text(encoding="utf-8").strip()).resolve()
    return git_dir


def _resolve(ref: str, git_dir: Path, common_dir: Path) -> str | None:
    for folder in (git_dir, common_dir):
        loose = folder / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip() or None
    packed = common_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8").splitlines():
            commit, _, name = line.partition(" ")
            if name.strip() == ref:
                return commit
    return None
