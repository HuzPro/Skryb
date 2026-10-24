import os
import subprocess
from pathlib import Path

import pytest

from skryb.core.checkout_update import pull_main

# Real git against a bare repository standing in for GitHub.
pytestmark = pytest.mark.integration

_IDENTITY = {
    "GIT_AUTHOR_NAME": "Tester",
    "GIT_AUTHOR_EMAIL": "tester@example.com",
    "GIT_COMMITTER_NAME": "Tester",
    "GIT_COMMITTER_EMAIL": "tester@example.com",
}


def _git(folder: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-c", "init.defaultBranch=main", "-c", "core.autocrlf=false", *arguments],
        cwd=folder, capture_output=True, text=True, encoding="utf-8", check=True,
        env={**os.environ, **_IDENTITY},
    ).stdout.strip()


def _commit(folder: Path, name: str, text: str) -> str:
    (folder / name).write_text(text, encoding="utf-8")
    _git(folder, "add", name)
    _git(folder, "commit", "-q", "-m", f"change {name}")
    return _git(folder, "rev-parse", "HEAD")


@pytest.fixture()
def github(tmp_path: Path) -> Path:
    """A bare repository standing in for GitHub, with one commit on main."""
    remote = tmp_path / "github.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", "-q", str(remote), str(seed))
    _commit(seed, "app.py", "print('first')\n")
    _git(seed, "push", "-q", "origin", "main")
    return remote


def _clone(github: Path, folder: Path) -> Path:
    _git(github.parent, "clone", "-q", str(github), str(folder))
    return folder


def _merge_on_github(github: Path, name: str = "naïve_ünïcode.py") -> str:
    """Someone merges a change on GitHub after the checkout was made."""
    other = _clone(github, github.parent / "other")
    commit = _commit(other, name, "# 日本語のコメント\nprint('merged')\n")
    _git(other, "push", "-q", "origin", "main")
    return commit


class TestPullingMain:
    def test_a_clean_checkout_of_main_catches_up_with_github(self, github: Path) -> None:
        checkout = _clone(github, github.parent / "app")
        merged = _merge_on_github(github)

        assert pull_main(checkout) is None
        assert _git(checkout, "rev-parse", "HEAD") == merged
        assert (checkout / "naïve_ünïcode.py").read_text(encoding="utf-8").endswith("'merged')\n")

    def test_a_checkout_already_current_is_left_as_it_is(self, github: Path) -> None:
        checkout = _clone(github, github.parent / "app")
        before = _git(checkout, "rev-parse", "HEAD")

        assert pull_main(checkout) is None
        assert _git(checkout, "rev-parse", "HEAD") == before

    def test_untracked_files_such_as_the_venv_do_not_stop_the_pull(self, github: Path) -> None:
        checkout = _clone(github, github.parent / "app")
        (checkout / "venv").mkdir()
        (checkout / "venv" / "pyvenv.cfg").write_text("home = C:\\Python312\n", encoding="utf-8")
        merged = _merge_on_github(github)

        assert pull_main(checkout) is None
        assert _git(checkout, "rev-parse", "HEAD") == merged

    def test_uncommitted_changes_leave_the_checkout_untouched(self, github: Path) -> None:
        checkout = _clone(github, github.parent / "app")
        before = _git(checkout, "rev-parse", "HEAD")
        (checkout / "app.py").write_text("print('work in progress')\n", encoding="utf-8")
        _merge_on_github(github)

        assert pull_main(checkout) == "this folder has uncommitted changes"
        assert _git(checkout, "rev-parse", "HEAD") == before
        assert (checkout / "app.py").read_text(encoding="utf-8") == "print('work in progress')\n"

    def test_a_checkout_on_another_branch_is_left_alone(self, github: Path) -> None:
        checkout = _clone(github, github.parent / "app")
        _git(checkout, "switch", "-q", "-c", "fix/pill-text")
        before = _git(checkout, "rev-parse", "HEAD")
        _merge_on_github(github)

        assert pull_main(checkout) == "this folder is on fix/pill-text, not main"
        assert _git(checkout, "rev-parse", "HEAD") == before

    def test_local_commits_github_lacks_make_git_refuse_and_nothing_moves(
        self, github: Path,
    ) -> None:
        checkout = _clone(github, github.parent / "app")
        local = _commit(checkout, "local.py", "print('only here')\n")
        _merge_on_github(github)

        reason = pull_main(checkout)

        assert reason is not None and reason.startswith("git said")
        assert _git(checkout, "rev-parse", "HEAD") == local

    def test_an_unreachable_github_says_why_and_changes_nothing(self, github: Path) -> None:
        checkout = _clone(github, github.parent / "app")
        before = _git(checkout, "rev-parse", "HEAD")
        _git(checkout, "remote", "set-url", "origin", str(github.parent / "gone.git"))

        reason = pull_main(checkout)

        assert reason is not None and reason.startswith("git said")
        assert _git(checkout, "rev-parse", "HEAD") == before

    def test_a_folder_that_is_not_a_checkout_is_left_alone(self, tmp_path: Path) -> None:
        assert pull_main(tmp_path) == "this folder isn't a git checkout"
