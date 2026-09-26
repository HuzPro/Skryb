"""Answer ROCm's GPU-family question before ``import torch`` asks it."""

import importlib
import logging
import os
from collections.abc import Sequence
from types import ModuleType

_logger = logging.getLogger(__name__)

_TARGET_FAMILY_ENV = "ROCM_SDK_TARGET_FAMILY"
_FAMILY_SUFFIXES = ("-all", "-dgpu", "-igpu", "-dcgpu")
_OFFLOAD_ARCH_TIMEOUT_SECONDS = 5.0


def prepare_rocm_environment() -> None:
    """Set ``ROCM_SDK_TARGET_FAMILY`` so importing torch runs no child process.

    On ROCm builds ``import torch`` asks ``rocm_sdk`` for the GPU family,
    and unless this variable is set ``rocm_sdk`` runs ``offload-arch``
    without ``CREATE_NO_WINDOW``. Under the windowless launcher that child
    gets a console window of its own, which flashes on screen. Answering
    first, silently, means it never runs. Best effort: never raises.
    """
    if os.environ.get(_TARGET_FAMILY_ENV) is not None:
        return
    try:
        family = _detect_target_family()
    except Exception:
        _logger.debug("ROCm target family left to rocm_sdk", exc_info=True)
        return
    if family is not None:
        os.environ[_TARGET_FAMILY_ENV] = family
        _logger.debug("ROCm target family: %s", family)


def target_family_for(
    offload_arch_output: str, available: Sequence[str],
) -> str | None:
    """The first available family matching an arch ``offload-arch`` printed.

    Mirrors ``rocm_sdk._dist_info.discover_current_target_family``: an arch
    such as ``gfx1201`` belongs to family ``gfx120X`` under one of the
    suffixes, and an exact arch name is itself a family.
    """
    for arch in dict.fromkeys(offload_arch_output.split()):
        family = arch[:-1] + "X"
        for candidate in (*(family + s for s in _FAMILY_SUFFIXES), arch):
            if candidate in available:
                return candidate
    return None


def _detect_target_family() -> str | None:
    dist_info = _rocm_dist_info()
    if dist_info is None:
        return None
    available = tuple(str(f) for f in dist_info.AVAILABLE_TARGET_FAMILIES)
    if len(available) == 1:
        return available[0]
    return target_family_for(_offload_arch_output(), available)


def _rocm_dist_info() -> ModuleType | None:
    try:
        return importlib.import_module("rocm_sdk._dist_info")
    except ImportError:
        return None


def _offload_arch_output() -> str:
    # Only multi-family installs get here, so these stay off startup.
    import shutil
    import subprocess
    import sysconfig

    search_path = os.pathsep.join(
        (sysconfig.get_path("scripts"), os.environ.get("PATH", "")),
    )
    executable = shutil.which("offload-arch", path=search_path)
    if executable is None:
        return ""
    return subprocess.run(
        [executable],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=_OFFLOAD_ARCH_TIMEOUT_SECONDS,
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ).stdout
