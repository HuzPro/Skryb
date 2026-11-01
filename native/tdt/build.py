"""Build the CPU TDT decoder: tdt_cpu.dll with MSVC on Windows, libtdt_cpu.so with gcc elsewhere. No GPU needed.

    python native/tdt/build.py

MSVC is found through vswhere and set up by hand, so it works where vcvars64.bat is missing. Floating-point
contraction stays off so the decoder's rounding matches the NumPy reference in the tests.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "tdt_cpu.c"
BUILD = HERE / "build"
_VSWHERE = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
_KITS = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Windows Kits" / "10"


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    if sys.platform == "win32":
        output = BUILD / "tdt_cpu.dll"
        environment, compiler = _msvc()
        subprocess.run(
            [str(compiler), "/nologo", "/O2", "/LD", "/fp:precise", str(SOURCE), f"/Fe:{output}", f"/Fo:{BUILD}\\"],
            check=True, env=environment, cwd=BUILD,
        )
    else:
        output = BUILD / "libtdt_cpu.so"
        subprocess.run(
            ["gcc", "-O3", "-shared", "-fPIC", "-pthread", "-ffp-contract=off", "-fvisibility=hidden", str(SOURCE), "-o", str(output), "-lm"],
            check=True,
        )
    print(f"built {output}")


def _msvc() -> tuple[dict[str, str], Path]:
    """PATH, INCLUDE and LIB for the newest x64 MSVC and Windows SDK installed, and its cl.exe.

    Windows finds a program on the caller's PATH, not the child's, so cl.exe is named in full.
    """
    install = subprocess.run(
        [str(_VSWHERE), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    msvc = max((Path(install) / "VC" / "Tools" / "MSVC").iterdir())
    sdk = max(path.name for path in (_KITS / "Include").iterdir() if (path / "ucrt").is_dir())
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join([str(msvc / "bin" / "Hostx64" / "x64"), str(_KITS / "bin" / sdk / "x64"), environment["PATH"]])
    environment["INCLUDE"] = os.pathsep.join(
        [str(msvc / "include")] + [str(_KITS / "Include" / sdk / part) for part in ("ucrt", "um", "shared")]
    )
    environment["LIB"] = os.pathsep.join(
        [str(msvc / "lib" / "x64")] + [str(_KITS / "Lib" / sdk / part / "x64") for part in ("ucrt", "um")]
    )
    return environment, msvc / "bin" / "Hostx64" / "x64" / "cl.exe"


if __name__ == "__main__":
    main()
