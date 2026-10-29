"""The latest.json a release publishes, read back the way the installed app reads it."""

import hashlib
import json
import os

import pytest

from release.latest_manifest import latest_manifest, write_manifest
from skryb.core.release_update import is_newer, release_from_github

# About the size of a real installer's first few MB; random so no two runs share a digest.
_INSTALLER = os.urandom(3 * 1024 * 1024 + 17)


@pytest.mark.parametrize("version", ["0.7.0", "0.10.0", "1.0.0", "10.20.300"])
def test_the_file_a_release_publishes_is_one_the_installed_app_accepts(tmp_path, version):
    (tmp_path / f"Skryb-Setup-{version}.exe").write_bytes(_INSTALLER)

    written = write_manifest(version, tmp_path)
    release = release_from_github(json.loads(written.read_text(encoding="utf-8")))

    assert written.name == "latest.json"
    assert release is not None
    assert release.version == version
    assert release.size == len(_INSTALLER)
    assert release.sha256 == hashlib.sha256(_INSTALLER).hexdigest()
    assert release.url == f"https://github.com/HuzPro/Skryb/releases/download/v{version}/Skryb-Setup-{version}.exe"
    assert is_newer(release.version, "0.6.0")


def test_an_empty_installer_is_never_offered(tmp_path):
    installer = tmp_path / "Skryb-Setup-0.7.0.exe"
    installer.write_bytes(b"")

    assert release_from_github(latest_manifest("0.7.0", installer)) is None


def test_a_release_whose_installer_is_missing_stops_before_publishing(tmp_path):
    (tmp_path / "Skryb-Setup-0.6.9.exe").write_bytes(_INSTALLER)

    with pytest.raises(FileNotFoundError):
        write_manifest("0.7.0", tmp_path)
    assert not (tmp_path / "latest.json").exists()
