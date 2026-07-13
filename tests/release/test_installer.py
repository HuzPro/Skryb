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
    build_iscc_args,
    cmd_shim_text,
    iss_defines,
    notice_text,
)


class TestIssDefines:
    def test_carries_the_version_payload_and_wheel(self):
        defines = iss_defines("0.2.0", Path("payload"), "voicetotext-0.2.0-py3-none-any.whl")

        assert defines["AppVersion"] == "0.2.0"
        assert defines["PayloadDir"] == str(Path("payload"))
        assert defines["WheelFile"] == "voicetotext-0.2.0-py3-none-any.whl"

    def test_maps_each_target_to_its_requirements_file(self):
        defines = iss_defines("0.2.0", Path("payload"), "app.whl")

        assert defines["ReqCpu"] == "requirements-cpu.txt"
        assert defines["ReqCuda"] == "requirements-nvidia.txt"
        assert defines["ReqRocm"] == "requirements-amd.txt"


class TestBuildIsccArgs:
    def test_turns_each_define_into_a_d_flag(self):
        args = build_iscc_args(
            Path("setup.iss"), Path("dist"), "VoiceToText-Setup-0.2.0", {"AppVersion": "0.2.0"}
        )

        assert "/DAppVersion=0.2.0" in args

    def test_sets_the_output_dir_and_basename(self):
        args = build_iscc_args(
            Path("setup.iss"), Path("out"), "VoiceToText-Setup-0.2.0", {}
        )

        assert f"/O{Path('out')}" in args
        assert "/FVoiceToText-Setup-0.2.0" in args

    def test_passes_the_script_path_last(self):
        args = build_iscc_args(Path("installer/VoiceToText.iss"), Path("dist"), "x", {})

        assert args[-1] == str(Path("installer/VoiceToText.iss"))


class TestLauncherShim:
    def test_invokes_the_console_entry_point_with_forwarded_args(self):
        shim = cmd_shim_text()

        assert "@echo off" in shim
        assert "voicetotext.exe" in shim
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
