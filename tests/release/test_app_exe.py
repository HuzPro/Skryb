"""Tests for the details compiled into Skryb.exe and skryb-cli.exe.

Compiling them needs Visual Studio and the bundled Python, so that runs in the installer build,
which also starts each exe and checks the Python it brings up. What is tested here is the data
that build is given: the version numbers, the resources, and the stack size read from Python's exe.
"""

import struct
from pathlib import Path, PureWindowsPath

import pytest

from release.app_exe import CONSOLE_EXE, WINDOWED_EXE, dlls_needed_to_start, file_version, resource_script, stack_size
from release.errors import InstallerBuildError


class TestTheVersionWindowsStores:
    @pytest.mark.parametrize(
        ("version", "numbers"),
        [
            ("0.18.1", (0, 18, 1, 0)),
            ("1.0", (1, 0, 0, 0)),
            ("2", (2, 0, 0, 0)),
            ("10.20.30.40", (10, 20, 30, 40)),
            ("0.0.0", (0, 0, 0, 0)),
            ("65535.65535.65535", (65535, 65535, 65535, 0)),
        ],
    )
    def test_a_release_version_becomes_four_numbers(self, version, numbers):
        assert file_version(version) == numbers

    @pytest.mark.parametrize("version", ["0.19.0rc1", "0.19.0.dev3+g1a2b3c4", "0.19.0-beta", "0.19.0 ", "0.19.0.1.2"])
    def test_whatever_follows_the_numbers_is_left_out(self, version):
        assert file_version(version)[:3] == (0, 19, 0)

    @pytest.mark.parametrize("version", ["", " ", "v0.18.1", "rc1", ".5", "٣.١", "-1.0"])
    def test_a_version_that_does_not_start_with_a_number_is_refused(self, version):
        with pytest.raises(InstallerBuildError, match="doesn't start with a number"):
            file_version(version)

    def test_a_number_windows_cannot_store_is_refused(self):
        with pytest.raises(InstallerBuildError, match="65535"):
            file_version("0.65536.0")


class TestTheResourcesOfAnExe:
    def test_task_manager_is_given_the_apps_name_and_the_release_version(self):
        script = resource_script("0.18.1", WINDOWED_EXE, Path("skryb.ico"), Path("python.manifest"))

        assert 'VALUE "FileDescription", "Skryb"' in script
        assert 'VALUE "ProductName", "Skryb"' in script
        assert 'VALUE "FileVersion", "0.18.1"' in script
        assert 'VALUE "ProductVersion", "0.18.1"' in script
        assert "FILEVERSION 0,18,1,0" in script
        assert "PRODUCTVERSION 0,18,1,0" in script

    @pytest.mark.parametrize(("exe", "internal"), [(WINDOWED_EXE, "Skryb"), (CONSOLE_EXE, "skryb-cli")])
    def test_each_exe_is_named_as_itself_and_described_as_skryb(self, exe, internal):
        script = resource_script("0.18.1", exe, Path("skryb.ico"), Path("python.manifest"))

        assert f'VALUE "OriginalFilename", "{exe}"' in script
        assert f'VALUE "InternalName", "{internal}"' in script
        assert 'VALUE "FileDescription", "Skryb"' in script

    def test_a_pre_release_keeps_its_full_name_in_the_text_and_only_numbers_in_the_header(self):
        script = resource_script("0.19.0rc1", WINDOWED_EXE, Path("skryb.ico"), Path("python.manifest"))

        assert 'VALUE "FileVersion", "0.19.0rc1"' in script
        assert "FILEVERSION 0,19,0,0" in script

    def test_the_icon_and_the_manifest_are_compiled_in_from_any_folder(self):
        folder = r"C:\Users\Zoë O'Hara\Skryb builds (new)"

        script = resource_script("0.18.1", WINDOWED_EXE, PureWindowsPath(folder, "skryb.ico"), PureWindowsPath(folder, "python.manifest"))

        # A backslash is doubled, or the resource compiler would read it as an escape.
        assert r'''1 ICON "C:\\Users\\Zoë O'Hara\\Skryb builds (new)\\skryb.ico"''' in script
        assert r'''1 24 "C:\\Users\\Zoë O'Hara\\Skryb builds (new)\\python.manifest"''' in script
        assert script.startswith("#pragma code_page(65001)")


def _exe_reserving(stack: int, header_at: int = 0x80) -> bytes:
    """The start of a 64-bit program file whose header asks for ``stack`` bytes of stack."""
    dos = bytearray(header_at)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, header_at)
    optional = bytearray(240)
    struct.pack_into("<H", optional, 0, 0x20B)
    struct.pack_into("<QQ", optional, 72, stack, 4096)
    return bytes(dos) + b"PE\0\0" + bytes(20) + bytes(optional)


class TestTheStackAnExeReserves:
    @pytest.mark.parametrize("stack", [2_000_000, 1_048_576, 8_388_608, 0])
    def test_it_is_read_from_the_programs_header(self, tmp_path, stack):
        exe = tmp_path / "pythonw.exe"
        exe.write_bytes(_exe_reserving(stack))

        assert stack_size(exe) == stack

    def test_the_header_is_found_wherever_the_file_puts_it(self, tmp_path):
        exe = tmp_path / "python.exe"
        exe.write_bytes(_exe_reserving(2_000_000, header_at=0x108) + bytes(50_000))

        assert stack_size(exe) == 2_000_000


_SKRYB_EXE_DEPENDENTS = """
Dump of file C:/payload/python/Skryb.exe

File Type: EXECUTABLE IMAGE

  Image has the following dependencies:

    api-ms-win-crt-heap-l1-1-0.dll
    api-ms-win-crt-runtime-l1-1-0.dll
    KERNEL32.dll
    api-ms-win-crt-string-l1-1-0.dll

  Image has the following delay load dependencies:

    python312.dll

  Summary

       11000 .data
        3000 .text
"""

_PYTHONW_EXE_DEPENDENTS = """
Dump of file C:/payload/python/pythonw.exe

File Type: EXECUTABLE IMAGE

  Image has the following dependencies:

    python312.dll
    VCRUNTIME140.dll
    api-ms-win-crt-runtime-l1-1-0.dll
    KERNEL32.dll

  Summary

        1000 .data
"""


class TestTheDllsAnExeNeedsToStart:
    def test_a_dll_loaded_only_on_first_use_is_not_one_of_them(self):
        # The settings window never runs Python, so it never loads python312.dll and
        # an update can replace that file while the window stays open.
        assert dlls_needed_to_start(_SKRYB_EXE_DEPENDENTS) == [
            "api-ms-win-crt-heap-l1-1-0.dll", "api-ms-win-crt-runtime-l1-1-0.dll", "KERNEL32.dll",
            "api-ms-win-crt-string-l1-1-0.dll",
        ]

    def test_an_exe_linked_like_pythons_own_needs_the_install_folders_dlls(self):
        assert dlls_needed_to_start(_PYTHONW_EXE_DEPENDENTS)[:2] == ["python312.dll", "VCRUNTIME140.dll"]

    @pytest.mark.parametrize("listing", ["", "dumpbin : fatal error LNK1181: cannot open input file", "\n\n  Summary\n"])
    def test_a_listing_that_names_no_dll_gives_none(self, listing):
        assert dlls_needed_to_start(listing) == []
