"""Where the native engines are on this machine: ``qasr.dll`` (Qwen3-ASR), ``npk.dll`` (Parakeet TDT) and
``nd.dll`` (the meeting diarizer).

Kept free of heavy imports: the engine registry asks at startup.
"""

import importlib.util
import os
from collections.abc import Mapping
from pathlib import Path

DLL_ENV = "QASR_DLL"
# The release ships it in the wheel next to this module; a source checkout
# builds it with native/engine/build.sh.
SHIPPED_DLL = Path(__file__).with_name("qasr.dll")
DEV_BUILD_DLL = Path(__file__).resolve().parents[2] / "native" / "engine" / "build" / "qasr.dll"


PARAKEET_DLL_ENV = "NPK_DLL"
SHIPPED_PARAKEET_DLL = Path(__file__).with_name("npk.dll")
DEV_BUILD_PARAKEET_DLL = Path(__file__).resolve().parents[2] / "native" / "parakeet" / "build" / "npk.dll"

DIARIZER_DLL_ENV = "ND_DLL"
SHIPPED_DIARIZER_DLL = Path(__file__).with_name("nd.dll")
DEV_BUILD_DIARIZER_DLL = Path(__file__).resolve().parents[2] / "native" / "diar" / "build" / "nd.dll"


def native_engine_dll(environ: Mapping[str, str] = os.environ) -> Path | None:
    """The DLL to load, or None when there is none. ``QASR_DLL`` wins and never falls back."""
    return _find_dll(environ, DLL_ENV, SHIPPED_DLL, DEV_BUILD_DLL)


def native_parakeet_dll(environ: Mapping[str, str] = os.environ) -> Path | None:
    """The same for ``npk.dll``. ``NPK_DLL`` wins and never falls back."""
    return _find_dll(environ, PARAKEET_DLL_ENV, SHIPPED_PARAKEET_DLL, DEV_BUILD_PARAKEET_DLL)


def native_diarizer_dll(environ: Mapping[str, str] = os.environ) -> Path | None:
    """The same for ``nd.dll``. ``ND_DLL`` wins and never falls back."""
    return _find_dll(environ, DIARIZER_DLL_ENV, SHIPPED_DIARIZER_DLL, DEV_BUILD_DIARIZER_DLL)


def rocm_runtime_dir() -> Path | None:
    """The HIP runtime the DLLs link against, from the ROCm SDK wheel, found without importing it."""
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))) / "bin"


def _find_dll(environ: Mapping[str, str], variable: str, shipped: Path, dev_build: Path) -> Path | None:
    configured = environ.get(variable, "").strip()
    candidates = [Path(configured)] if configured else [shipped, dev_build]
    return next((path for path in candidates if path.is_file()), None)
