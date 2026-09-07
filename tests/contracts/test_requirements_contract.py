"""Contract: every one-shot install file pulls PyTorch.

The platform requirements files (`requirements-nvidia.txt`,
`requirements-cpu.txt`, `requirements-amd.txt`,
`requirements-amd-linux.txt`) are the ONLY place torch-family wheels
get pinned. `requirements.txt` / `requirements-all.txt` deliberately
omit torch so a pre-existing custom build is left untouched, so they
are not one-shot files and are not checked here.

This contract fails the build if any one-shot file or the canonical
`pyproject.toml` dependency list drops the engine backbone.
"""

import re
import tomllib
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]

_REQUIRED_TORCH_PACKAGES = frozenset({"torch"})

_ONE_SHOT_INSTALL_FILES = (
    "requirements-nvidia.txt",
    "requirements-cpu.txt",
    "requirements-amd.txt",
    "requirements-amd-linux.txt",
)

_PACKAGE_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def _names_from_requirement_lines(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = _PACKAGE_NAME.match(line)
        if match:
            names.add(_normalize(match.group(1)))
    return names


def _missing_required_packages(declared: set[str]) -> set[str]:
    return {_normalize(pkg) for pkg in _REQUIRED_TORCH_PACKAGES} - declared


def _packages_in_requirements_file(filename: str) -> set[str]:
    text = (_REPO_ROOT / filename).read_text(encoding="utf-8")
    return _names_from_requirement_lines(text.splitlines())


def _packages_in_pyproject_dependencies() -> set[str]:
    pyproject = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return _names_from_requirement_lines(pyproject["project"]["dependencies"])


@pytest.mark.parametrize("filename", _ONE_SHOT_INSTALL_FILES)
def test_one_shot_install_file_declares_every_required_torch_package(
    filename: str,
) -> None:
    missing = _missing_required_packages(_packages_in_requirements_file(filename))
    assert not missing, (
        f"{filename} is a one-shot install file but omits {sorted(missing)}; "
        f"users on this platform cannot install all required deps from it."
    )


def test_pyproject_dependencies_declare_every_required_torch_package() -> None:
    missing = _missing_required_packages(_packages_in_pyproject_dependencies())
    assert not missing, (
        f"pyproject.toml [project].dependencies omits {sorted(missing)}; "
        f"the canonical dependency list must declare every torch-family "
        f"runtime dependency."
    )


def test_parakeet_is_a_core_dependency_and_not_an_extra() -> None:
    pyproject = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = _names_from_requirement_lines(
        pyproject["project"]["dependencies"]
    )
    requirements = _packages_in_requirements_file("requirements.txt")

    assert "nano-parakeet" in dependencies
    assert "nano-parakeet" in requirements
    assert "parakeet" not in pyproject["project"]["optional-dependencies"]


def test_editable_install_declares_console_free_gui_launcher() -> None:
    pyproject = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["project"]["gui-scripts"]["voicetotext-gui"] == (
        "voicetotext.__main__:main"
    )


def test_requirements_include_every_non_torch_runtime_dependency() -> None:
    required = _packages_in_pyproject_dependencies() - {"torch"}

    assert required <= _packages_in_requirements_file("requirements.txt")
