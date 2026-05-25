"""Shared classifier for engine-backend ImportError surfaces.

Each adapter's ``_import_backend`` wraps its lazy backend import in a
try/except so a missing or broken backend surfaces as
:class:`EngineDependencyMissingError` instead of a raw ``ImportError``.
The challenge is the message — telling the user "pip install
transformers" when transformers IS installed (but a submodule import
exploded because of a torch ABI mismatch) wastes their time and hides
the real cause.

This module's :func:`make_engine_dependency_error` checks whether the
top-level package can be found by ``importlib.util.find_spec``. If yes,
the failure is downstream (submodule import / ABI skew) and the
returned error names that and quotes the underlying message. If no,
the package is genuinely absent and the install hint is returned.
"""

import importlib.util

from voicetotext.core.exceptions import EngineDependencyMissingError


def make_engine_dependency_error(
    *,
    package_name: str,
    install_hint: str,
    underlying: ImportError,
) -> EngineDependencyMissingError:
    if _package_can_be_located(package_name):
        return EngineDependencyMissingError(
            f"{package_name} is installed but failed to load: {underlying}. "
            f"This usually indicates an ABI mismatch between {package_name} "
            f"and torch (a common case on AMD Windows ROCm) — pin both to "
            f"the versions listed in INSTALL.md."
        )
    return EngineDependencyMissingError(install_hint)


def _package_can_be_located(package_name: str) -> bool:
    try:
        return importlib.util.find_spec(package_name) is not None
    except (ImportError, ValueError):
        return False
