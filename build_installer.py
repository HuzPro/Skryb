"""Build the VoiceToText Windows installer (setup .exe).

Usage::

    python build_installer.py --python-dir <standalone CPython 3.12 dir>
    python build_installer.py --python-dir ... --version 0.2.0 --iscc "C:\\...\\ISCC.exe"

Stages the installer payload (a standalone CPython runtime, the app wheel,
and the requirements files) and compiles it with Inno Setup into
``dist/VoiceToText-Setup-<version>.exe``. PyTorch is not bundled; the
installer pip-installs the hardware-appropriate wheels on the user's
machine at setup time, which keeps the published asset small.

The standalone CPython runtime is supplied via ``--python-dir`` (CI
downloads a python-build-standalone release; locally, point it at any
relocatable CPython 3.12). Inno Setup must be installed, or its compiler
path passed via ``--iscc``.
"""

import argparse
import sys
from pathlib import Path

from release.errors import ReleaseError
from release.installer import compile_installer, locate_iscc, stage_payload
from release.version import resolve_version


_REPO_ROOT = Path(__file__).resolve().parent
_ISS_PATH = _REPO_ROOT / "installer" / "VoiceToText.iss"


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        setup = _build(args)
    except ReleaseError as error:
        print(f"Installer build aborted: {error}", file=sys.stderr)
        return 1
    print(f"Built {setup}")
    return 0


def _build(args: argparse.Namespace) -> Path:
    version = resolve_version(args.version, _REPO_ROOT / "pyproject.toml")
    payload_dir = _REPO_ROOT / "build" / "installer-payload"
    output_dir = _REPO_ROOT / "dist"
    output_dir.mkdir(parents=True, exist_ok=True)

    wheel_filename = stage_payload(_REPO_ROOT, payload_dir, Path(args.python_dir), version)
    iscc = locate_iscc(args.iscc)
    return compile_installer(_ISS_PATH, payload_dir, version, wheel_filename, output_dir, iscc)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the VoiceToText Windows installer.")
    parser.add_argument(
        "--python-dir",
        required=True,
        help="Path to a standalone CPython 3.12 runtime to bundle.",
    )
    parser.add_argument(
        "--version",
        default="",
        help="Release version. Defaults to the version in pyproject.toml.",
    )
    parser.add_argument(
        "--iscc",
        default="",
        help="Path to ISCC.exe. Auto-detected from the default install if omitted.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
