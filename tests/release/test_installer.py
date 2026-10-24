"""Tests for assembling the Inno Setup installer.

The installer is compiled by Inno Setup's command-line compiler (ISCC),
parameterised with ``/D`` defines so one ``.iss`` script serves every
build. Constructing those defines and the ISCC argument vector is pure
data work and is tested here; copying the payload and invoking ISCC are
thin I/O steps exercised in CI. The launcher shim and the runtime NOTICE
text are also pinned down, since both ship verbatim to users.
"""

from pathlib import Path

import pytest

from release.errors import InstallerBuildError
from release.installer import (
    _require_standalone_runtime,
    _stage_wheel_source,
    build_iscc_args,
    cmd_shim_text,
    iss_defines,
    notice_text,
)


class TestIssDefines:
    def test_carries_the_version_payload_and_wheel(self):
        defines = iss_defines("0.2.0", Path("payload"), "skryb-0.2.0-py3-none-any.whl")

        assert defines["AppVersion"] == "0.2.0"
        assert defines["PayloadDir"] == str(Path("payload"))
        assert defines["WheelFile"] == "skryb-0.2.0-py3-none-any.whl"

    def test_maps_each_target_to_its_requirements_file(self):
        defines = iss_defines("0.2.0", Path("payload"), "app.whl")

        assert defines["ReqCpu"] == "requirements-cpu.txt"
        assert defines["ReqCuda"] == "requirements-nvidia.txt"
        assert defines["ReqRocm"] == "requirements-amd.txt"


class TestBuildIsccArgs:
    def test_turns_each_define_into_a_d_flag(self):
        args = build_iscc_args(
            Path("setup.iss"), Path("dist"), "Skryb-Setup-0.2.0", {"AppVersion": "0.2.0"}
        )

        assert "/DAppVersion=0.2.0" in args

    def test_sets_the_output_dir_and_basename(self):
        args = build_iscc_args(
            Path("setup.iss"), Path("out"), "Skryb-Setup-0.2.0", {}
        )

        assert f"/O{Path('out')}" in args
        assert "/FSkryb-Setup-0.2.0" in args

    def test_passes_the_script_path_last(self):
        args = build_iscc_args(Path("installer/Skryb.iss"), Path("dist"), "x", {})

        assert args[-1] == str(Path("installer/Skryb.iss"))


class TestLauncherShim:
    def test_invokes_the_console_entry_point_with_forwarded_args(self):
        shim = cmd_shim_text()

        assert "@echo off" in shim
        assert "skryb.exe" in shim
        assert "%*" in shim


class TestValidatingThePythonRuntime:
    def test_accepts_a_standalone_runtime(self, tmp_path: Path):
        (tmp_path / "python.exe").write_text("", encoding="utf-8")
        (tmp_path / "Lib").mkdir()

        _require_standalone_runtime(tmp_path)  # does not raise

    def test_rejects_a_virtualenv_scripts_folder(self, tmp_path: Path):
        # A venv's Scripts folder has python.exe but no sibling Lib/.
        (tmp_path / "python.exe").write_text("", encoding="utf-8")

        with pytest.raises(InstallerBuildError):
            _require_standalone_runtime(tmp_path)


class TestRuntimeNotice:
    def test_states_the_python_runtime_and_runtime_installed_packages(self):
        notice = notice_text("0.2.0")

        assert "0.2.0" in notice
        assert "Python Software Foundation" in notice
        assert "THIRD-PARTY-NOTICES" in notice
        assert "LICENSE" in notice

    def test_states_the_app_is_apache_licensed_not_proprietary(self):
        notice = notice_text("0.2.0")

        assert "Apache License" in notice
        assert "Huzaifa Tatari" in notice
        assert "proprietary" not in notice.lower()


def _fake_repo(root: Path, *, ui_built: bool = True) -> Path:
    repo = root / "repo"
    (repo / "skryb").mkdir(parents=True)
    (repo / "skryb" / "__init__.py").write_text("", encoding="utf-8")
    for name in ("pyproject.toml", "LICENSE.txt", "NOTICE"):
        (repo / name).write_text(name, encoding="utf-8")
    if ui_built:
        exe = repo / "ui" / "src-tauri" / "target" / "release" / "skryb-ui.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"MZ settings window")
    (root / "scratch").mkdir()
    return repo


class TestStageWheelSource:
    def test_stages_package_metadata_and_license_files(self, tmp_path):
        # pyproject.toml declares `license-files = ["LICENSE.txt", "NOTICE"]`,
        # so the staged wheel source must carry them or the build fails.
        repo = _fake_repo(tmp_path)

        source = _stage_wheel_source(repo, tmp_path / "scratch")

        assert (source / "pyproject.toml").is_file()
        assert (source / "LICENSE.txt").is_file()
        assert (source / "NOTICE").is_file()
        assert (source / "skryb" / "__init__.py").is_file()

    def test_puts_the_built_settings_window_where_the_installed_app_looks(self, tmp_path):
        repo = _fake_repo(tmp_path)

        source = _stage_wheel_source(repo, tmp_path / "scratch")

        shipped = source / "skryb" / "ui-bin" / "skryb-ui.exe"
        assert shipped.read_bytes() == b"MZ settings window"

    def test_refuses_to_build_an_installer_without_the_settings_window(self, tmp_path):
        repo = _fake_repo(tmp_path, ui_built=False)

        with pytest.raises(InstallerBuildError, match=r"build_ui\.ps1"):
            _stage_wheel_source(repo, tmp_path / "scratch")
