"""Write ``latest.json``, the file the installed app reads to find the newest release.

GitHub's API answers 60 anonymous requests an hour per internet connection,
shared with every other program on it, so the app's updater reads this file
through the release's ordinary download link instead, which has no such
limit. It has the shape of the API's ``releases/latest`` answer, cut down to
what ``skryb.core.release_update.release_from_github`` reads.

``python -m release.latest_manifest 0.7.0 artifacts`` writes
``artifacts/latest.json`` for ``artifacts/Skryb-Setup-0.7.0.exe``.
"""

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY = "HuzPro/Skryb"
MANIFEST_NAME = "latest.json"


def latest_manifest(version: str, installer: Path) -> dict[str, object]:
    with installer.open("rb") as handle:
        sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "tag_name": f"v{version}",
        "assets": [{
            "name": installer.name,
            "size": installer.stat().st_size,
            "digest": f"sha256:{sha256}",
            "browser_download_url": f"https://github.com/{REPOSITORY}/releases/download/v{version}/{installer.name}",
        }],
    }


def write_manifest(version: str, folder: Path) -> Path:
    manifest = latest_manifest(version, folder / f"Skryb-Setup-{version}.exe")
    target = folder / MANIFEST_NAME
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


if __name__ == "__main__":
    version, folder = sys.argv[1:]
    print(write_manifest(version, Path(folder)))
