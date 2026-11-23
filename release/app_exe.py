"""Build Skryb.exe and skryb-cli.exe, the programs an installed Skryb runs as.

Windows names a process after its program file, so a Skryb started through ``pythonw.exe`` shows
as "Python". These two are small programs of Skryb's own (``native/host/skryb_host.c``) that sit
beside the bundled Python and run it in their own process: Skryb.exe for the shortcuts, and
skryb-cli.exe for the ``skryb`` command, which keeps the terminal.

Each is made to match the Python program it stands in for: the same stack size and the same
manifest, both read from the bundled runtime, so a new CPython build cannot leave them behind.
"""

import importlib.util
import json
import re
import struct
import subprocess
import tempfile
from pathlib import Path

from release import APP_NAME
from release.errors import InstallerBuildError

WINDOWED_EXE = f"{APP_NAME}.exe"
CONSOLE_EXE = "skryb-cli.exe"

_SOURCE = Path("native") / "host" / "skryb_host.c"
_ICON = Path("brand") / "skryb.ico"
_MSVC_FINDER = Path("native") / "tdt" / "build.py"
# (the exe, the Python program it stands in for, extra compiler flags, the linker's subsystem)
_BUILDS = (
    (WINDOWED_EXE, "pythonw.exe", (), "WINDOWS"),
    (CONSOLE_EXE, "python.exe", ("/DSKRYB_CONSOLE",), "CONSOLE"),
)
_LARGEST_VERSION_PART = 65535
_STARTS_IN_SECONDS = 60
# Win32: LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_IMAGE_RESOURCE, RT_MANIFEST, and an exe's own manifest.
_LOAD_AS_DATA = 0x2 | 0x20
_MANIFEST_RESOURCE = 24
_APP_MANIFEST_ID = 1
# A stand-in `skryb` module: it reports how Python was started, so the check never opens the app.
_REPORTER = "import json, sys\nopen(sys.argv[1], 'w', encoding='utf-8').write(json.dumps([sys.executable, sys.argv[2:]]))\nsys.exit(7)\n"


def file_version(version: str) -> tuple[int, int, int, int]:
    """The four numbers Windows stores for ``version``; a pre-release suffix has no place in them."""
    numbers = re.match(r"[0-9]+(?:\.[0-9]+){0,3}", version)
    if numbers is None:
        raise InstallerBuildError(f"version {version!r} doesn't start with a number")
    parts = [int(part) for part in numbers.group().split(".")]
    if max(parts) > _LARGEST_VERSION_PART:
        raise InstallerBuildError(f"version {version!r} has a part above {_LARGEST_VERSION_PART}, the most Windows stores")
    return tuple(parts + [0] * (4 - len(parts)))  # type: ignore[return-value]


def resource_script(version: str, exe_name: str, icon: Path, manifest: Path) -> str:
    """The resources compiled into an exe: its icon, its manifest and the details Windows shows.

    Task Manager names a process by ``FileDescription``, so that is the app's name.
    """
    numbers = ",".join(str(part) for part in file_version(version))
    details = {
        "CompanyName": APP_NAME,
        "FileDescription": APP_NAME,
        "FileVersion": version,
        "InternalName": Path(exe_name).stem,
        "OriginalFilename": exe_name,
        "ProductName": APP_NAME,
        "ProductVersion": version,
    }
    values = "\n".join(f'            VALUE "{name}", "{_quoted(value)}"' for name, value in details.items())
    return f"""#pragma code_page(65001)
1 ICON "{_quoted(str(icon))}"
{_APP_MANIFEST_ID} {_MANIFEST_RESOURCE} "{_quoted(str(manifest))}"
1 VERSIONINFO
FILEVERSION {numbers}
PRODUCTVERSION {numbers}
FILEOS 0x40004L
FILETYPE 0x1L
BEGIN
    BLOCK "StringFileInfo"
    BEGIN
        BLOCK "040904B0"
        BEGIN
{values}
        END
    END
    BLOCK "VarFileInfo"
    BEGIN
        VALUE "Translation", 0x409, 1200
    END
END
"""


def stack_size(exe: Path) -> int:
    """The stack ``exe`` reserves for its main thread, from its header."""
    header = exe.read_bytes()[:4096]
    optional_header = struct.unpack_from("<I", header, 0x3C)[0] + 24
    return struct.unpack_from("<Q", header, optional_header + 72)[0]


def manifest_of(exe: Path) -> bytes:
    """The manifest compiled into ``exe``, byte for byte. Windows only, like the build itself."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LoadLibraryExW.restype = kernel.FindResourceW.restype = kernel.LoadResource.restype = wintypes.HANDLE
    kernel.LockResource.restype = ctypes.c_void_p
    kernel.LoadLibraryExW.argtypes = (wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD)
    kernel.FindResourceW.argtypes = (wintypes.HANDLE, wintypes.LPVOID, wintypes.LPVOID)
    for function in (kernel.LoadResource, kernel.SizeofResource):
        function.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel.LockResource.argtypes = kernel.FreeLibrary.argtypes = (wintypes.HANDLE,)
    module = kernel.LoadLibraryExW(str(exe), None, _LOAD_AS_DATA)
    found = kernel.FindResourceW(module, _APP_MANIFEST_ID, _MANIFEST_RESOURCE) if module else None
    if not found:
        raise InstallerBuildError(f"{exe} has no manifest to copy (Windows error {ctypes.get_last_error()})")
    data = ctypes.string_at(kernel.LockResource(kernel.LoadResource(module, found)), kernel.SizeofResource(module, found))
    kernel.FreeLibrary(module)
    return data


def build_app_exes(repo_root: Path, python_dir: Path, version: str) -> None:
    """Build both exes into ``python_dir``, beside the bundled Python they run, and check they start it."""
    environment, compiler = _load_msvc_finder(repo_root)()
    with tempfile.TemporaryDirectory() as scratch:
        for exe_name, interpreter, flags, subsystem in _BUILDS:
            work = Path(scratch) / Path(exe_name).stem
            work.mkdir()
            manifest = work / "python.manifest"
            manifest.write_bytes(manifest_of(python_dir / interpreter))
            (work / "app.rc").write_text(
                resource_script(version, exe_name, repo_root / _ICON, manifest), encoding="utf-8",
            )
            _run(["rc.exe", "/nologo", "/fo", "app.res", "app.rc"], environment, work)
            _run(
                [
                    str(compiler), "/nologo", "/O2", "/MD", "/W3", *flags, f"/I{python_dir / 'include'}",
                    str(repo_root / _SOURCE), "app.res", f"/Fe:{python_dir / exe_name}",
                    "/link", f"/LIBPATH:{python_dir / 'libs'}", f"/SUBSYSTEM:{subsystem}", "/MANIFEST:NO",
                    f"/STACK:{stack_size(python_dir / interpreter)}",
                ],
                environment, work,
            )
            _check_it_starts_python(python_dir / exe_name, python_dir / interpreter, work)


def _check_it_starts_python(exe: Path, interpreter: Path, work: Path) -> None:
    (work / "skryb").mkdir()
    (work / "skryb" / "__main__.py").write_text(_REPORTER, encoding="utf-8")
    report = work / "started.json"
    arguments = ["--page", "Größe ✓ and spaces", ""]
    # -m looks in the working folder first, so this runs the stand-in even where Skryb is installed.
    result = subprocess.run([str(exe), str(report), *arguments], cwd=work, timeout=_STARTS_IN_SECONDS)
    started = json.loads(report.read_text(encoding="utf-8")) if report.exists() else None
    if result.returncode != 7 or started != [str(interpreter), arguments]:
        raise InstallerBuildError(
            f"{exe.name} did not start the bundled Python as {interpreter.name} does: "
            f"exit code {result.returncode}, reported {started!r}"
        )


def _load_msvc_finder(repo_root: Path):
    spec = importlib.util.spec_from_file_location("skryb_native_tdt_build", repo_root / _MSVC_FINDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._msvc


def _run(command: list[str], environment: dict[str, str], work: Path) -> None:
    # Windows finds a program on the caller's PATH, not the child's, so the SDK tools are looked up here.
    program = _on_path(command[0], environment["PATH"])
    subprocess.run([program, *command[1:]], check=True, env=environment, cwd=work)


def _on_path(program: str, path: str) -> str:
    if Path(program).is_absolute():
        return program
    for folder in path.split(";"):
        if (Path(folder) / program).is_file():
            return str(Path(folder) / program)
    raise InstallerBuildError(f"{program} was not found; install the Windows SDK with Visual Studio's C++ tools")


def _quoted(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '""')
