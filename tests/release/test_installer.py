"""Tests for assembling the Inno Setup installer.

The installer is compiled by Inno Setup's command-line compiler (ISCC),
parameterised with ``/D`` defines so one ``.iss`` script serves every
build. Constructing those defines and the ISCC argument vector is pure
data work and is tested here; copying the payload and invoking ISCC are
thin I/O steps exercised in CI. The launcher shim and the runtime NOTICE
text are also pinned down, since both ship verbatim to users.
"""

import fnmatch
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


def _fake_repo(root: Path, *, ui_built: bool = True, engine_built: bool = True) -> Path:
    repo = root / "repo"
    (repo / "skryb").mkdir(parents=True)
    (repo / "skryb" / "__init__.py").write_text("", encoding="utf-8")
    for name in ("pyproject.toml", "LICENSE.txt", "NOTICE"):
        (repo / name).write_text(name, encoding="utf-8")
    if ui_built:
        exe = repo / "ui" / "src-tauri" / "target" / "release" / "skryb-ui.exe"
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"MZ settings window")
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
        nemo_speech = repo / "native" / "nemo_speech" / "build"
        (nemo_speech / "licenses" / "third_party" / "ggml").mkdir(parents=True)
        (nemo_speech / "nemo_speech_asr_c.dll").write_bytes(b"MZ nemo speech")
        (nemo_speech / "ggml-cpu.dll").write_bytes(b"MZ ggml")
        (nemo_speech / "licenses" / "LICENSE").write_text("Apache License 2.0", encoding="utf-8")
        (nemo_speech / "licenses" / "third_party" / "ggml" / "LICENSE").write_text("MIT", encoding="utf-8")
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

        shipped = source / "skryb" / "ui-bin" / "skryb-ui.exe"
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



class TestNvidiasCpuDiarizer:
    def test_ships_its_dlls_and_licences_as_one_folder_beside_the_native_engines(self, tmp_path):
        repo = _fake_repo(tmp_path)

        source = _stage_wheel_source(repo, tmp_path / "scratch")

        shipped = source / "skryb" / "transcription" / "nemo-speech"
        assert (shipped / "nemo_speech_asr_c.dll").read_bytes() == b"MZ nemo speech"
        assert (shipped / "ggml-cpu.dll").read_bytes() == b"MZ ggml"
        assert (shipped / "licenses" / "LICENSE").read_text(encoding="utf-8") == "Apache License 2.0"
        assert (shipped / "licenses" / "third_party" / "ggml" / "LICENSE").is_file()

    def test_refuses_to_build_an_installer_without_it(self, tmp_path):
        repo = _fake_repo(tmp_path)
        (repo / "native" / "nemo_speech" / "build" / "nemo_speech_asr_c.dll").unlink()

        with pytest.raises(InstallerBuildError, match=r"native/nemo_speech/fetch\.py"):
            _stage_wheel_source(repo, tmp_path / "scratch")


def _package_data_ships(path: str, patterns: list[str]) -> bool:
    """Whether a setuptools package-data glob ships *path*: ``*`` stays within one folder."""
    parts = path.split("/")
    return any(
        len(pattern.split("/")) == len(parts) and all(map(fnmatch.fnmatchcase, parts, pattern.split("/")))
        for pattern in patterns
    )


def test_the_wheel_declares_every_file_of_nvidias_cpu_diarizer_that_fetch_keeps() -> None:
    import tomllib

    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = pyproject["tool"]["setuptools"]["package-data"]["skryb"]
    kept = [
        "nemo_speech_asr_c.dll", "ggml-cpu.dll", "msvcp140.dll", "licenses/LICENSE", "licenses/NOTICE",
        "licenses/THIRD_PARTY_NOTICES.md", "licenses/third_party/ggml/LICENSE", "licenses/third_party/llama.cpp/LICENSE",
        "licenses/third_party/vcpkg/abseil/LICENSE", "licenses/third_party/vcpkg/utf8-range/LICENSE",
    ]

    unshipped = [name for name in kept if not _package_data_ships(f"transcription/nemo-speech/{name}", patterns)]

    assert unshipped == []
