"""Tests for assembling the Inno Setup installer.

The installer is compiled by Inno Setup's command-line compiler (ISCC),
parameterised with ``/D`` defines so one ``.iss`` script serves every
build. Constructing those defines and the ISCC argument vector is pure
data work and is tested here; copying the payload and invoking ISCC are
thin I/O steps exercised in CI. The launcher shim and the runtime NOTICE
text are also pinned down, since both ship verbatim to users.
"""

import tomllib
from pathlib import Path

import pytest

from release.errors import InstallerBuildError
from release.installer import (
    _UI_SHELL_IN_WHEEL,
    _require_standalone_runtime,
    _stage_wheel_source,
    build_iscc_args,
    cmd_shim_text,
    iss_defines,
    notice_text,
)
from skryb.ui.launcher import SETTINGS_WINDOW_FLAG


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
    def test_the_skryb_command_runs_skrybs_own_console_program_with_what_was_typed(self):
        shim = cmd_shim_text()

        assert shim.splitlines() == ["@echo off", r'"%~dp0..\python\skryb-cli.exe" %*']
        # cmd.exe needs CRLF line ends.
        assert shim.endswith("\r\n") and "\n" not in shim.replace("\r\n", "")


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


def _fake_repo(root: Path, *, ui_built: bool = True, engine_built: bool = True) -> Path:
    repo = root / "repo"
    (repo / "skryb").mkdir(parents=True)
    (repo / "skryb" / "__init__.py").write_text("", encoding="utf-8")
    for name in ("pyproject.toml", "LICENSE.txt", "NOTICE"):
        (repo / name).write_text(name, encoding="utf-8")
    if ui_built:
        window = repo / "ui" / "src-tauri" / "target" / "release" / "skryb_ui.dll"
        window.parent.mkdir(parents=True)
        window.write_bytes(b"MZ settings window")
    if engine_built:
        dll = repo / "native" / "engine" / "build" / "qasr.dll"
        dll.parent.mkdir(parents=True)
        dll.write_bytes(b"MZ native engine")
        parakeet = repo / "native" / "parakeet" / "build" / "npk.dll"
        parakeet.parent.mkdir(parents=True)
        parakeet.write_bytes(b"MZ native parakeet")
        diarizer = repo / "native" / "diar" / "build" / "nd.dll"
        diarizer.parent.mkdir(parents=True)
        diarizer.write_bytes(b"MZ native diarizer")
        for amd in (dll, parakeet, diarizer):
            amd.with_name(f"{amd.stem}_cuda.dll").write_bytes(b"MZ nvidia " + amd.name.encode())
        decoder = repo / "native" / "tdt" / "build" / "tdt_cpu.dll"
        decoder.parent.mkdir(parents=True)
        decoder.write_bytes(b"MZ cpu decoder")
    (repo / "skryb" / "transcription").mkdir()
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

        shipped = source / "skryb" / "ui-bin" / "skryb_ui.dll"
        assert shipped.read_bytes() == b"MZ settings window"

    def test_puts_the_native_engine_where_the_installed_app_looks(self, tmp_path):
        repo = _fake_repo(tmp_path)

        source = _stage_wheel_source(repo, tmp_path / "scratch")

        assert (source / "skryb" / "transcription" / "qasr.dll").read_bytes() == b"MZ native engine"
        assert (source / "skryb" / "transcription" / "npk.dll").read_bytes() == b"MZ native parakeet"
        assert (source / "skryb" / "transcription" / "nd.dll").read_bytes() == b"MZ native diarizer"
        assert (source / "skryb" / "transcription" / "tdt_cpu.dll").read_bytes() == b"MZ cpu decoder"

    def test_puts_each_engines_nvidia_build_beside_it(self, tmp_path):
        repo = _fake_repo(tmp_path)

        source = _stage_wheel_source(repo, tmp_path / "scratch")

        for name in ("qasr", "npk", "nd"):
            assert (source / "skryb" / "transcription" / f"{name}_cuda.dll").read_bytes() == b"MZ nvidia " + f"{name}.dll".encode()

    def test_refuses_to_build_an_installer_without_an_nvidia_build(self, tmp_path):
        repo = _fake_repo(tmp_path)
        (repo / "native" / "diar" / "build" / "nd_cuda.dll").unlink()

        with pytest.raises(InstallerBuildError, match=r"nd_cuda\.dll is missing.*native/cuda/build\.py"):
            _stage_wheel_source(repo, tmp_path / "scratch")

    def test_refuses_to_build_an_installer_without_the_native_parakeet_engine(self, tmp_path):
        repo = _fake_repo(tmp_path)
        (repo / "native" / "parakeet" / "build" / "npk.dll").unlink()

        with pytest.raises(InstallerBuildError, match=r"native/parakeet/build\.sh"):
            _stage_wheel_source(repo, tmp_path / "scratch")

    def test_refuses_to_build_an_installer_without_the_npu_engines_cpu_decoder(self, tmp_path):
        repo = _fake_repo(tmp_path)
        (repo / "native" / "tdt" / "build" / "tdt_cpu.dll").unlink()

        with pytest.raises(InstallerBuildError, match=r"native/tdt/build\.py"):
            _stage_wheel_source(repo, tmp_path / "scratch")

    def test_refuses_to_build_an_installer_without_the_native_engine(self, tmp_path):
        repo = _fake_repo(tmp_path, engine_built=False)

        with pytest.raises(InstallerBuildError, match=r"native/engine/build\.sh"):
            _stage_wheel_source(repo, tmp_path / "scratch")

    def test_refuses_to_build_an_installer_without_the_settings_window(self, tmp_path):
        repo = _fake_repo(tmp_path, ui_built=False)

        with pytest.raises(InstallerBuildError, match=r"build_ui\.ps1"):
            _stage_wheel_source(repo, tmp_path / "scratch")


_SETUP_SCRIPT = Path(__file__).resolve().parents[2] / "installer" / "Skryb.iss"


def _packaged_patterns() -> set[str]:
    """The files pyproject.toml puts in the wheel beside the code, with the globs it uses filled in."""
    with (_SETUP_SCRIPT.parents[1] / "pyproject.toml").open("rb") as handle:
        patterns = tomllib.load(handle)["tool"]["setuptools"]["package-data"]["skryb"]
    return {"skryb/" + pattern.replace("ui-bin/*.dll", "ui-bin/skryb_ui.dll") for pattern in patterns}


def _setup_script_section(name: str) -> list[str]:
    """The entries of one section of the Setup script, comments left out."""
    section = _SETUP_SCRIPT.read_text(encoding="utf-8").split(f"[{name}]", 1)[1].split("\n[", 1)[0]
    return [line.strip() for line in section.splitlines() if line.strip() and not line.lstrip().startswith(";")]


def _entry(line: str) -> dict[str, str]:
    """``Name: "a"; Filename: "b"`` as a dict."""
    return {key.strip(): value.strip().strip('"') for key, value in (part.split(":", 1) for part in line.split(";"))}


class TestTheSetupScript:
    def test_every_shortcut_that_opens_the_app_starts_skrybs_own_exe(self):
        shortcuts = [_entry(line) for line in _setup_script_section("Icons")]
        opens_the_app = [shortcut for shortcut in shortcuts if "Uninstall" not in shortcut["Name"]]

        assert [shortcut["Name"] for shortcut in opens_the_app] == [r"{group}\{#AppName}", r"{autodesktop}\{#AppName}"]
        for shortcut in opens_the_app:
            # Through pythonw.exe, Windows lists the running app as "Python".
            assert shortcut["Filename"] == r"{app}\python\{#AppName}.exe"
            assert "Parameters" not in shortcut
            assert shortcut["WorkingDir"] == "{app}"
            # The file Skryb repaints in the colour chosen in Settings.
            assert shortcut["IconFilename"] == r"{app}\skryb.ico"

    def test_the_app_reopened_after_an_update_is_skrybs_own_exe(self):
        reopened = [_entry(line) for line in _setup_script_section("Run")]

        assert len(reopened) == 1
        assert reopened[0]["Filename"] == r"{app}\python\{#AppName}.exe"
        assert "Parameters" not in reopened[0]
        assert reopened[0]["WorkingDir"] == "{app}"
        # A silent in-app update must reopen Skryb, so this never gets skipifsilent.
        assert reopened[0]["Flags"].split() == ["nowait", "postinstall"]

    def test_nothing_setup_links_or_starts_runs_the_app_through_python(self):
        entries = _setup_script_section("Icons") + _setup_script_section("Run")

        assert not [line for line in entries if "python.exe" in line.lower() or "pythonw.exe" in line.lower()]

    def test_shortcuts_left_by_earlier_versions_are_pointed_at_skrybs_own_exe_on_every_install(self):
        code = _SETUP_SCRIPT.read_text(encoding="utf-8").split("[Code]", 1)[1]
        after_copying = code.split("if CurStep = ssPostInstall then", 1)[1].split("if CurStep = ssDone then", 1)[0]

        assert "PointOldShortcutsAtSkryb();" in after_copying
        # WScript.Shell reads a shortcut as empty when its path has a character outside the system
        # code page, so one named "Skryb ✓.lnk" would be skipped. Explorer's own object reads it.
        assert "CreateOleObject('Shell.Application')" in code
        assert "WScript.Shell')" not in code
        assert code.isascii(), "Setup reads the script as ANSI: a character outside ASCII in it is a gamble"
        # The desktop, the Start menu with its Startup folder, and where taskbar pins are kept.
        for folder in ("{userdesktop}", "{userprograms}", r"{userappdata}\Microsoft\Internet Explorer\Quick Launch"):
            assert f"ExpandConstant('{folder}')" in code

    def test_the_running_skryb_exe_is_moved_aside_before_setup_looks_for_programs_to_close(self):
        code = _SETUP_SCRIPT.read_text(encoding="utf-8").split("[Code]", 1)[1]
        # Setup calls PrepareToInstall first, then asks Windows what is using the files it will copy.
        before_the_copy = code.split("function PrepareToInstall(", 1)[1].split("end;", 1)[0]
        when_setup_ends = code.split("procedure DeinitializeSetup();", 1)[1].split("end;", 1)[0]

        assert "SetRunningAppExeAside();" in before_the_copy
        # An install that stopped before the copy must not leave the shortcuts without their exe.
        assert "SettleAppExeSetAside();" in when_setup_ends

    def test_an_old_settings_window_is_found_whichever_program_it_runs_as(self):
        code = _SETUP_SCRIPT.read_text(encoding="utf-8").split("[Code]", 1)[1]
        closing = code.split("procedure CloseOldSettingsWindows();", 1)[1].split("end;", 1)[0]

        # Up to 0.18 the window was skryb-ui.exe; now it is Skryb.exe started with this flag.
        assert "''skryb-ui.exe''" in closing
        assert "''{#AppName}.exe''" in closing
        assert f"'' {SETTINGS_WINDOW_FLAG}''" in closing

    def test_skryb_exe_looks_for_the_settings_window_where_the_wheel_puts_it(self):
        host = (_SETUP_SCRIPT.parents[1] / "native" / "host" / "skryb_host.c").read_text(encoding="utf-8")
        beside_the_exe = "Lib/site-packages/" + _UI_SHELL_IN_WHEEL

        assert 'L"' + beside_the_exe.replace("/", r"\\") + '"' in host
        assert f'L"{SETTINGS_WINDOW_FLAG}"' in host
        assert _UI_SHELL_IN_WHEEL in _packaged_patterns()

    def test_the_skryb_it_reopens_after_an_update_can_follow_the_users_own_folder_links(self):
        directives = _setup_script_section("Setup")

        # Left on (Inno Setup's default since 6.7.0), the reopened app inherits the guard and
        # every model folder that is a junction to another drive fails to open with error 448.
        assert "RedirectionGuard=no" in directives
        assert "PrivilegesRequired=lowest" in directives
