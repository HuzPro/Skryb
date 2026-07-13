"""Resolve the version a release is cut as.

Precedence is explicit-over-declared: a version supplied on the command
line (or, in CI, parsed from the release commit message) wins; otherwise
the build falls back to the canonical version in ``pyproject.toml``. An
empty or whitespace-only explicit value means "not supplied".
"""

import tomllib
from pathlib import Path

from release.errors import VersionResolutionError


def resolve_version(explicit: str, pyproject_path: Path) -> str:
    """Return the release version, preferring ``explicit`` when given."""
    chosen = explicit.strip()
    if chosen:
        return chosen
    return read_pyproject_version(pyproject_path)


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
