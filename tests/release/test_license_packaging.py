"""The gated-model licence text must survive packaging.

The installer installs the wheel, so a runtime resource that is not
declared as package data produces an installed app that cannot display
or hash the terms it is gating on. These tests take the real staging
step the installer uses, build a real wheel from it, and check the bytes
that come out the far end against the audited digest.
"""

import hashlib
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from release import installer
from release.installer import _stage_wheel_source


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESOURCE = Path("voicetotext/licenses/crisperwhisper-nc.md")

# Fetched from both pinned CrisperWhisper commits, where the bytes are
# identical. Changing this constant means the app now displays different
# terms, which retires every acceptance recorded against the old ones.
_AUDITED_DIGEST = (
    "42f43bf4dc72ef422f91f6ca143f4514e21a68b353940d99091c1addcd0aee79"
)
_AUDITED_SIZE = 17784

_BUILD_WHEEL = (
    "import os, sys; os.chdir(sys.argv[1]); "
    "from setuptools import build_meta; "
    "print(build_meta.build_wheel(sys.argv[2]))"
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="module")
def staged_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Staging requires the built settings window, which CI never builds.
    stand_in = tmp_path_factory.mktemp("ui") / "voicetotext-ui.exe"
    stand_in.write_bytes(b"MZ")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(installer, "_UI_SHELL_BUILD", stand_in)
        return _stage_wheel_source(_REPO_ROOT, tmp_path_factory.mktemp("stage"))


@pytest.fixture(scope="module")
def built_wheel(
    staged_source: Path, tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    output = tmp_path_factory.mktemp("wheel")
    subprocess.run(
        [sys.executable, "-c", _BUILD_WHEEL, str(staged_source), str(output)],
        check=True,
        capture_output=True,
    )
    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    return wheels[0]


class TestShippedLicenceInRepo:
    def test_should_carry_the_audited_bytes(self) -> None:
        payload = (_REPO_ROOT / _RESOURCE).read_bytes()

        assert len(payload) == _AUDITED_SIZE
        assert _digest(payload) == _AUDITED_DIGEST


class TestPyprojectDeclaration:
    def test_should_declare_the_licence_directory_as_package_data(self) -> None:
        pyproject = tomllib.loads(
            (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )

        package_data = pyproject["tool"]["setuptools"]["package-data"]

        assert "licenses/*.md" in package_data["voicetotext"]


class TestStagedInstallerPayload:
    def test_should_carry_the_licence_unchanged(self, staged_source: Path) -> None:
        payload = (staged_source / _RESOURCE).read_bytes()

        assert _digest(payload) == _AUDITED_DIGEST


class TestBuiltWheel:
    def test_should_contain_the_licence_at_its_import_path(
        self, built_wheel: Path,
    ) -> None:
        with zipfile.ZipFile(built_wheel) as wheel:
            names = wheel.namelist()

        assert _RESOURCE.as_posix() in names

    def test_should_contain_the_licence_bytes_unchanged(
        self, built_wheel: Path,
    ) -> None:
        with zipfile.ZipFile(built_wheel) as wheel:
            payload = wheel.read(_RESOURCE.as_posix())

        assert len(payload) == _AUDITED_SIZE
        assert _digest(payload) == _AUDITED_DIGEST
