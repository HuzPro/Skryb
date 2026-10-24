"""Resolve the version a release is cut as.

``pyproject.toml`` is the one source: the wheel, so the installed app,
reports that version. A version supplied on the command line (or, in CI,
parsed from the release commit message) must match it, because an
installer that reports an older version than its tag would have the app's
updater offer the same update forever. An empty or whitespace-only
explicit value means "not supplied".
"""

import tomllib
from pathlib import Path

from release.errors import VersionResolutionError


def resolve_version(explicit: str, pyproject_path: Path) -> str:
    """Return the release version, refusing an ``explicit`` one ``pyproject.toml`` doesn't declare."""
    declared = read_pyproject_version(pyproject_path)
    chosen = explicit.strip().removeprefix("v")
    if chosen and chosen != declared:
        raise VersionResolutionError(
            f"release version {chosen} doesn't match pyproject.toml's {declared}; "
            f"set version = \"{chosen}\" in pyproject.toml in the release commit"
        )
    return declared


def read_pyproject_version(pyproject_path: Path) -> str:
    """Return the ``[project] version`` declared in ``pyproject_path``."""
    project = _read_project_table(pyproject_path)
    version = str(project.get("version", "")).strip()
    if not version:
        raise VersionResolutionError(
            f"no [project] version declared in {pyproject_path}"
        )
    return version


def _read_project_table(pyproject_path: Path) -> dict[str, object]:
    with pyproject_path.open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project", {})
    if not isinstance(project, dict):
        raise VersionResolutionError(
            f"[project] table missing or malformed in {pyproject_path}"
        )
    return project
