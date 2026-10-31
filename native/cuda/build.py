"""Builds the NVIDIA engines: qasr_cuda.dll, npk_cuda.dll, cw_cuda.dll and nd_cuda.dll, from the same sources as
the AMD ones. One DLL per engine covers the RTX 30 (sm_86), 40 (sm_89) and 50 (sm_120) series, with PTX for
newer cards; the CUDA runtime is linked in, so users need only the NVIDIA driver (R525+, R570+ on an RTX 50).

Needs the toolkit from fetch_toolkit.py (or CUDA_PATH) and Visual Studio 2022's C++ build tools; no GPU.

    python native/cuda/build.py [engine ...]    # engine: engine, parakeet, whisper, diar
"""

import os
import subprocess
import sys
from pathlib import Path

NATIVE = Path(__file__).resolve().parent.parent
ENGINES = {
    "engine": ("qasr_cuda.dll", ["kernels.hip", "engine.hip"]),
    "parakeet": ("npk_cuda.dll", ["../engine/kernels.hip", "pk_kernels.hip", "engine.hip"]),
    "whisper": ("cw_cuda.dll", ["../engine/kernels.hip", "wh_kernels.hip", "engine.hip"]),
    "diar": ("nd_cuda.dll", ["../engine/kernels.hip", "nd_kernels.hip", "engine.hip"]),
}
ARCHS = ["-gencode=arch=compute_86,code=sm_86", "-gencode=arch=compute_89,code=sm_89",
         "-gencode=arch=compute_120,code=sm_120", "-gencode=arch=compute_120,code=compute_120"]


def toolkit() -> Path:
    root = Path(os.environ.get("CUDA_PATH") or NATIVE / "cuda" / "build" / "toolkit")
    if not (root / "bin" / "nvcc.exe").is_file():
        raise SystemExit(f"no nvcc under {root}: run native/cuda/fetch_toolkit.py")
    return root


def msvc_environment() -> dict[str, str]:
    """This environment with Visual Studio's x64 C++ tools on it (vcvars64), unless cl.exe already is."""
    if subprocess.run(["where", "cl.exe"], capture_output=True).returncode == 0:
        return dict(os.environ)
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    install = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"], capture_output=True, text=True, check=True).stdout.strip()
    vcvars = Path(install) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    dump = subprocess.run(f'"{vcvars}" >nul && set', shell=True, capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in dump.splitlines() if "=" in line)


def main() -> None:
    nvcc = toolkit() / "bin" / "nvcc.exe"
    env = msvc_environment()
    builds = {}
    for name in sys.argv[1:] or ENGINES:
        dll, sources = ENGINES[name]
        folder = NATIVE / name
        (folder / "build").mkdir(exist_ok=True)
        # /MT: the static CUDA runtime is built against the static C runtime.
        builds[f"native/{name}/build/{dll}"] = subprocess.Popen(
            [str(nvcc), "-x", "cu", "-O3", "-std=c++20", "-shared", "-cudart", "static", "--threads", "0", *ARCHS,
             "-Xcompiler", "/EHsc /MT", "-o", f"build/{dll}", *sources],
            cwd=folder, env=env)
    failed = [dll for dll, build in builds.items() if build.wait() != 0]
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")
    print("built " + " ".join(builds))


if __name__ == "__main__":
    main()
