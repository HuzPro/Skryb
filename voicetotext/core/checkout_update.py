"""Fast-forward the source checkout from GitHub when nothing local is at stake.

The app runs straight from a git checkout of main, so this replaces the
manual ``git pull`` after a merge on GitHub. A checkout on another branch,
one with uncommitted changes, or one holding commits GitHub lacks is left
exactly as it is.
"""

import os
import subprocess
from pathlib import Path

from voicetotext.core.build_info import read_build

_BRANCH = "main"
# GitHub can take several seconds to answer from here; a stalled connection
# must still end.
_TIMEOUT_SECONDS = 60
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
# Never wait on a sign-in prompt nobody can see.
_NON_INTERACTIVE = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}


def pull_main(checkout: Path) -> str | None:
    """None once main matches GitHub's, otherwise why the checkout was left alone."""
    build = read_build(checkout)
    if build is None:
        return "this folder isn't a git checkout"
    if build.branch != _BRANCH:
        return f"this folder is on {build.branch or 'a detached commit'}, not main"
    try:
        return _fast_forward(checkout)
    except subprocess.TimeoutExpired:
        return "GitHub didn't answer in time"
    except subprocess.CalledProcessError as exc:
        return _git_refusal(exc)
    except OSError as exc:
        return f"git couldn't run: {exc}"


def _fast_forward(checkout: Path) -> str | None:
    if _git(checkout, "status", "--porcelain", "--untracked-files=no"):
        return "this folder has uncommitted changes"
    _git(checkout, "fetch", "--quiet", "origin", _BRANCH)
    _git(checkout, "merge", "--ff-only", "--quiet", "FETCH_HEAD")
    return None


def _git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_TIMEOUT_SECONDS,
        check=True,
        creationflags=_CREATE_NO_WINDOW,
        env={**os.environ, **_NON_INTERACTIVE},
    )
    return completed.stdout.strip()


def _git_refusal(exc: subprocess.CalledProcessError) -> str:
    said = (exc.stderr or "").strip().splitlines()
    return f"git said {said[-1]!r}" if said else "git failed"
