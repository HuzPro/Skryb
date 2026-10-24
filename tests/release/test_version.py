"""Tests for release-version resolution.

The release version comes from the commit message in CI and from a
``--version`` flag or nothing at all when building locally. Either way it
must be the version declared in ``pyproject.toml``, which is what the
installed app reports. An empty string models "not supplied" so no
``None`` ever crosses the boundary.
"""

from pathlib import Path

import pytest

from release.errors import VersionResolutionError
from release.version import read_pyproject_version, resolve_version


def _write_pyproject(directory: Path, version_line: str) -> Path:
    path = directory / "pyproject.toml"
    path.write_text(
        f'[project]\nname = "skryb"\n{version_line}\n',
        encoding="utf-8",
    )
    return path


class TestReadingThePyprojectVersion:
    def test_reads_the_declared_project_version(self, tmp_path: Path):
        pyproject = _write_pyproject(tmp_path, 'version = "0.1.0"')

        assert read_pyproject_version(pyproject) == "0.1.0"

    def test_raises_when_no_version_is_declared(self, tmp_path: Path):
        pyproject = _write_pyproject(tmp_path, 'description = "no version here"')

        with pytest.raises(VersionResolutionError):
            read_pyproject_version(pyproject)


class TestResolvingTheReleaseVersion:
    def test_accepts_an_explicit_version_that_matches_pyproject(self, tmp_path: Path):
        pyproject = _write_pyproject(tmp_path, 'version = "2.5.0"')

        assert resolve_version("2.5.0", pyproject) == "2.5.0"
        assert resolve_version("v2.5.0", pyproject) == "2.5.0"

    @pytest.mark.parametrize("explicit", ["2.5.0", "0.1.1", "0.1.0-rc1", "1.0.0"])
    def test_refuses_an_explicit_version_pyproject_does_not_declare(
        self, tmp_path: Path, explicit: str
    ):
        pyproject = _write_pyproject(tmp_path, 'version = "0.1.0"')

        with pytest.raises(VersionResolutionError, match="pyproject.toml"):
            resolve_version(explicit, pyproject)

    def test_falls_back_to_pyproject_when_no_explicit_version(
        self, tmp_path: Path
    ):
        pyproject = _write_pyproject(tmp_path, 'version = "0.1.0"')

        assert resolve_version("", pyproject) == "0.1.0"

    def test_treats_a_whitespace_only_explicit_version_as_absent(
        self, tmp_path: Path
    ):
        pyproject = _write_pyproject(tmp_path, 'version = "0.1.0"')

        assert resolve_version("   ", pyproject) == "0.1.0"

    def test_trims_surrounding_whitespace_from_an_explicit_version(
        self, tmp_path: Path
    ):
        pyproject = _write_pyproject(tmp_path, 'version = "2.5.0"')

        assert resolve_version("  2.5.0  ", pyproject) == "2.5.0"
