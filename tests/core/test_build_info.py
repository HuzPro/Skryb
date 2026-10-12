import os
from pathlib import Path

from voicetotext.core.build_info import (
    Build,
    RunningBuild,
    read_build,
    source_changed_since,
)

_FIRST = "fe502a2a2032de7e3a1a03ee3ddf9d25068b6e4f"
_SECOND = "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d"


def _checkout(root: Path, head: str = "ref: refs/heads/main\n") -> Path:
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head, encoding="utf-8")
    return git


def _point(git: Path, branch: str, commit: str) -> None:
    (git / "refs" / "heads" / branch).write_text(commit + "\n", encoding="utf-8")


class TestReadBuild:
    def test_reads_the_branch_and_its_commit(self, tmp_path: Path) -> None:
        _point(_checkout(tmp_path), "main", _FIRST)

        assert read_build(tmp_path) == Build(commit=_FIRST, branch="main")

    def test_reads_a_branch_kept_only_in_packed_refs(self, tmp_path: Path) -> None:
        git = _checkout(tmp_path)
        (git / "packed-refs").write_text(
            f"# pack-refs with: peeled fully-peeled sorted\n{_FIRST} refs/heads/main\n",
            encoding="utf-8",
        )

        assert read_build(tmp_path) == Build(commit=_FIRST, branch="main")

    def test_reads_a_detached_commit(self, tmp_path: Path) -> None:
        _checkout(tmp_path, head=_FIRST + "\n")

        assert read_build(tmp_path) == Build(commit=_FIRST)

    def test_follows_a_worktree_to_the_main_checkout_for_its_branch(
        self, tmp_path: Path,
    ) -> None:
        main_git = _checkout(tmp_path / "main")
        _point(main_git, "feature", _FIRST)
        worktree_git = main_git / "worktrees" / "feature"
        worktree_git.mkdir(parents=True)
        (worktree_git / "HEAD").write_text("ref: refs/heads/feature\n", encoding="utf-8")
        (worktree_git / "commondir").write_text("../..\n", encoding="utf-8")
        worktree = tmp_path / "feature"
        worktree.mkdir()
        (worktree / ".git").write_text(f"gitdir: {worktree_git}\n", encoding="utf-8")

        assert read_build(worktree) == Build(commit=_FIRST, branch="feature")

    def test_is_none_outside_a_checkout(self, tmp_path: Path) -> None:
        assert read_build(tmp_path) is None

    def test_is_none_for_a_branch_with_no_commit_yet(self, tmp_path: Path) -> None:
        _checkout(tmp_path)

        assert read_build(tmp_path) is None


class TestRunningBuild:
    def test_names_the_running_commit(self, tmp_path: Path) -> None:
        _point(_checkout(tmp_path), "main", _FIRST)

        assert RunningBuild(tmp_path).status() == "main fe502a2"

    def test_names_the_commit_a_restart_would_pick_up(self, tmp_path: Path) -> None:
        git = _checkout(tmp_path)
        _point(git, "main", _FIRST)
        running = RunningBuild(tmp_path)

        _point(git, "main", _SECOND)

        assert running.status() == "main fe502a2, restart for main 1a2b3c4"

    def test_says_nothing_when_not_run_from_a_checkout(self, tmp_path: Path) -> None:
        assert RunningBuild(tmp_path).status() == ""


_STARTED = 1_790_000_000.0


def _source_tree(root: Path, files: int) -> list[Path]:
    """A package of *files* modules spread over nested folders, all older than the start."""
    paths = []
    for index in range(files):
        folder = root / f"feature_{index % 40}" / f"part_{index % 7}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"module_{index}.py"
        path.write_text(f"VALUE = {index}\n", encoding="utf-8")
        os.utime(path, (_STARTED - 3600, _STARTED - 3600))
        paths.append(path)
    return paths


class TestSourceChangedSince:
    def test_thousands_of_files_written_before_the_start_mean_no_change(
        self, tmp_path: Path,
    ) -> None:
        _source_tree(tmp_path, 2_000)

        assert source_changed_since(_STARTED, tmp_path) is False

    def test_one_file_written_after_the_start_counts_however_deep_it_is(
        self, tmp_path: Path,
    ) -> None:
        paths = _source_tree(tmp_path, 2_000)
        os.utime(paths[1_337], (_STARTED + 1, _STARTED + 1))

        assert source_changed_since(_STARTED, tmp_path) is True

    def test_files_that_are_not_python_do_not_count(self, tmp_path: Path) -> None:
        _source_tree(tmp_path, 10)
        notes = tmp_path / "README.md"
        notes.write_text("notes", encoding="utf-8")
        os.utime(notes, (_STARTED + 60, _STARTED + 60))

        assert source_changed_since(_STARTED, tmp_path) is False
