"""Assemble the installer payload and compile it with Inno Setup.

The installer ships a standalone CPython runtime, the Skryb wheel,
and the three requirements files. It does *not* bundle PyTorch: the
matching wheel set is pip-installed on the user's machine at setup time
from its hardware target. That keeps the published asset small enough for
a GitHub Release, sidestepping the multi-GB bundle problem.

This module owns the build side: staging that payload and invoking ISCC.
The Inno Setup defines, the ISCC argument vector, the launcher shim, and
the runtime NOTICE are pure functions (unit-tested); copying files and
running subprocesses are thin, separately-named steps.
"""

import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from release import APP_NAME
from release.build_target import BuildTarget
from release.errors import InstallerBuildError


# The console entry point pip installs from ``[project.scripts]`` (lower
# case, the name users type), distinct from the display name APP_NAME.
_CLI_NAME = "skryb"

# Only these modules of the build-time ``release`` package are shipped, so
# the bundled Python can regenerate THIRD-PARTY-NOTICES.txt post-install.
_NOTICES_TOOL_MODULES = ("__init__.py", "notices.py")

# Where ISCC is usually installed, searched when no path is given. Covers
# the standard machine install and the per-user winget install.
_ISCC_DEFAULT_LOCATIONS = (
    r"%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe",
    r"%ProgramFiles%\Inno Setup 6\ISCC.exe",
    r"%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe",
)

# The settings window (Tauri) as scripts\build_ui.ps1 leaves it, and where it
# sits in the wheel: skryb.ui.launcher looks in ui-bin/ when installed.
_UI_SHELL_BUILD = Path("ui") / "src-tauri" / "target" / "release" / "skryb-ui.exe"
_UI_SHELL_IN_WHEEL = "skryb/ui-bin/skryb-ui.exe"


def iss_defines(version: str, payload_dir: Path, wheel_filename: str) -> dict[str, str]:
    """Build the ``/D`` defines the ``.iss`` script reads."""
    defines = {
        "AppVersion": version,
        "PayloadDir": str(payload_dir),
        "WheelFile": wheel_filename,
    }
    for target in BuildTarget:
        defines[_define_name(target)] = target.requirements_file
    return defines


def build_iscc_args(
    iss_path: Path,
    output_dir: Path,
    output_basename: str,
    defines: dict[str, str],
) -> list[str]:
    """Build the ISCC argument vector (everything after ``ISCC.exe``)."""
    args = [f"/D{name}={value}" for name, value in defines.items()]
    args.append(f"/O{output_dir}")
    args.append(f"/F{output_basename}")
    args.append(str(iss_path))
    return args


def cmd_shim_text() -> str:
    """Return the ``skryb.cmd`` launcher placed on the system PATH."""
    return "\r\n".join(
        [
            "@echo off",
            rf'"%~dp0..\python\Scripts\{_CLI_NAME}.exe" %*',
            "",
        ]
    )


def notice_text(version: str) -> str:
    """Return NOTICE.txt describing the bundled runtime and licensing."""
    return "\n".join(
        [
            f"{APP_NAME} {version} - Notices",
            "",
            "This application bundles a copy of the CPython runtime,",
            "distributed under the Python Software Foundation License. That",
            "license and the licenses of the components inside the CPython",
            "distribution are in the python\\ folder of this installation.",
            "",
            "The Python packages that power transcription (PyTorch,",
            "Transformers, and others) are downloaded from their official",
            "package indices during setup; they are not redistributed by",
            "this installer. Each installed package's license is recorded in",
            "its .dist-info folder under python\\Lib\\site-packages, and a",
            "consolidated summary is written to THIRD-PARTY-NOTICES.txt",
            "during installation.",
            "",
            f"{APP_NAME} is open source, licensed under the Apache License,",
            "Version 2.0; see LICENSE.txt. Copyright 2026 Huzaifa Tatari.",
            "",
        ]
    )


def _define_name(target: BuildTarget) -> str:
    return "Req" + target.slug.capitalize()


# --- payload staging and compilation (I/O) -------------------------------


def stage_payload(repo_root: Path, payload_dir: Path, python_dir: Path, version: str) -> str:
    """Stage everything the installer ships; return the wheel filename."""
    _require_standalone_runtime(python_dir)
    _reset_dir(payload_dir)
    wheel_filename = _build_app_wheel(python_dir, repo_root, payload_dir / "wheels")
    _copy_python_runtime(python_dir, payload_dir / "python")
    _copy_requirements(repo_root, payload_dir / "requirements")
    _copy_notices_tool(repo_root, payload_dir / "tools" / "release")
    _copy_license(repo_root, payload_dir / "LICENSE.txt")
    _write_text(payload_dir / "NOTICE.txt", notice_text(version))
    _write_text(payload_dir / "bin" / "skryb.cmd", cmd_shim_text())
    return wheel_filename


def compile_installer(
    iss_path: Path,
    payload_dir: Path,
    version: str,
    wheel_filename: str,
    output_dir: Path,
    iscc_path: Path,
) -> Path:
    """Compile the installer with ISCC; return the produced setup .exe."""
    output_basename = f"{APP_NAME}-Setup-{version}"
    defines = iss_defines(version, payload_dir, wheel_filename)
    args = build_iscc_args(iss_path, output_dir, output_basename, defines)
    subprocess.run([str(iscc_path), *args], check=True, cwd=str(payload_dir.parent))
    return output_dir / f"{output_basename}.exe"


def locate_iscc(explicit: str) -> Path:
    """Find the Inno Setup compiler, preferring an explicit path."""
    if explicit.strip():
        return _require_existing(Path(explicit.strip()))
    discovered = shutil.which("ISCC")
    if discovered:
        return Path(discovered)
    for candidate in _ISCC_DEFAULT_LOCATIONS:
        expanded = Path(os.path.expandvars(candidate))
        if expanded.exists():
            return expanded
    raise InstallerBuildError(
        "Inno Setup compiler (ISCC.exe) not found; install Inno Setup 6 or "
        "pass --iscc with its path"
    )


def _require_existing(path: Path) -> Path:
    if not path.exists():
        raise InstallerBuildError(f"ISCC not found at {path}")
    return path


def _build_app_wheel(python_dir: Path, repo_root: Path, wheels_dir: Path) -> str:
    # Build with the bundled runtime's own Python: it always ships pip, so
    # the interpreter that launched this script does not need pip itself.
    wheels_dir.mkdir(parents=True, exist_ok=True)
    python_exe = python_dir / "python.exe"
    with tempfile.TemporaryDirectory() as scratch:
        source = _stage_wheel_source(repo_root, Path(scratch))
        subprocess.run(
            [str(python_exe), "-m", "pip", "wheel", "--no-deps", "-w", str(wheels_dir), str(source)],
            check=True,
        )
    wheels = list(wheels_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise InstallerBuildError(
            f"expected exactly one app wheel in {wheels_dir}, found {len(wheels)}"
        )
    _require_ui_shell_in_wheel(wheels[0])
    return wheels[0].name


def _require_ui_shell_in_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        if _UI_SHELL_IN_WHEEL not in archive.namelist():
            raise InstallerBuildError(
                f"{wheel.name} is missing {_UI_SHELL_IN_WHEEL}; check "
                f"[tool.setuptools.package-data] in pyproject.toml"
            )


def _stage_wheel_source(repo_root: Path, scratch: Path) -> Path:
    # Build from a clean copy of just the project metadata and package, so
    # an in-repo venv or a stale build/ directory cannot pollute the wheel
    # (setuptools' scratch dir is also "build/") or slow the build down.
    # LICENSE.txt and NOTICE are declared in `license-files`, so setuptools
    # requires them next to pyproject.toml and embeds them in the wheel.
    source = scratch / "src"
    source.mkdir()
    for metadata_file in ("pyproject.toml", "LICENSE.txt", "NOTICE"):
        shutil.copyfile(repo_root / metadata_file, source / metadata_file)
    shutil.copytree(
        repo_root / "skryb",
        source / "skryb",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    _copy_ui_shell(repo_root, source / "skryb" / "ui-bin")
    return source


def _copy_ui_shell(repo_root: Path, destination: Path) -> None:
    built = repo_root / _UI_SHELL_BUILD
    if not built.exists():
        raise InstallerBuildError(
            f"the settings window is not built ({built} is missing); "
            f"run scripts\\build_ui.ps1 first"
        )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(built, destination / built.name)


def _require_standalone_runtime(python_dir: Path) -> None:
    # A standalone CPython has python.exe AND Lib/ at its root; a virtualenv
    # (or its Scripts folder) has python.exe but no sibling Lib/, so this
    # rejects the common mistake of pointing --python-dir at a venv.
    if not (python_dir / "python.exe").exists() or not (python_dir / "Lib").is_dir():
        raise InstallerBuildError(
            f"--python-dir must point at a standalone CPython 3.12 runtime "
            f"(a folder containing python.exe and Lib\\), not a virtualenv or "
            f"its Scripts folder. Got: {python_dir}. Download one from "
            f"https://github.com/astral-sh/python-build-standalone (the "
            f"install_only Windows x86_64 build)."
        )


def _copy_python_runtime(python_dir: Path, destination: Path) -> None:
    shutil.copytree(python_dir, destination)


def _copy_requirements(repo_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for target in BuildTarget:
        _copy_required_file(repo_root, target.requirements_file, destination)
    for shared in ("requirements-all.txt", "requirements.txt"):
        _copy_required_file(repo_root, shared, destination)


def _copy_required_file(repo_root: Path, name: str, destination: Path) -> None:
    source = repo_root / name
    if not source.exists():
        raise InstallerBuildError(f"required payload file missing: {source}")
    shutil.copyfile(source, destination / name)


def _copy_notices_tool(repo_root: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for module in _NOTICES_TOOL_MODULES:
        shutil.copyfile(repo_root / "release" / module, destination / module)


def _copy_license(repo_root: Path, destination: Path) -> None:
    source = repo_root / "LICENSE.txt"
    if not source.exists():
        raise InstallerBuildError(f"LICENSE.txt missing at {source}")
    shutil.copyfile(source, destination)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
