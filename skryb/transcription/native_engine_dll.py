"""Where ``qasr.dll``, the native Qwen3-ASR engine, is on this machine.

Kept free of heavy imports: the engine registry asks at startup.
"""

import os
from collections.abc import Mapping
from pathlib import Path

DLL_ENV = "QASR_DLL"
# The release ships it in the wheel next to this module; a source checkout
# builds it with native/engine/build.sh.
SHIPPED_DLL = Path(__file__).with_name("qasr.dll")
DEV_BUILD_DLL = Path(__file__).resolve().parents[2] / "native" / "engine" / "build" / "qasr.dll"


def native_engine_dll(environ: Mapping[str, str] = os.environ) -> Path | None:
    """The DLL to load, or None when there is none. ``QASR_DLL`` wins and never falls back."""
    configured = environ.get(DLL_ENV, "").strip()
    candidates = [Path(configured)] if configured else [SHIPPED_DLL, DEV_BUILD_DLL]
    return next((path for path in candidates if path.is_file()), None)
