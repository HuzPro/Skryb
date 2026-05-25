"""Tests for the shared backend-import error classifier.

The adapters wrap their lazy backend imports so a missing or broken
backend surfaces as :class:`EngineDependencyMissingError` instead of a
raw ``ImportError``. But the message must accurately distinguish
between two distinct failure modes:

* The package is **not installed** — pip install fixes it.
* The package **is** installed but a submodule import fails — usually
  an ABI mismatch with torch (e.g. AMD Windows ROCm torch lacks
  ``torch._C._distributed_c10d``, which transformers 5.x imports
  unconditionally). Telling the user "pip install transformers" here
  is misleading and wastes their time.
"""

from unittest.mock import MagicMock, patch

import pytest

from voicetotext.core.exceptions import EngineDependencyMissingError
from voicetotext.transcription.backend_imports import (
    make_engine_dependency_error,
)


class TestPackageMissing:
    def test_should_return_install_hint_when_package_is_not_installed(self) -> None:
        underlying = ImportError("No module named 'transformers'")

        with patch(
            "voicetotext.transcription.backend_imports."
            "importlib.util.find_spec",
            return_value=None,
        ):
            error = make_engine_dependency_error(
                package_name="transformers",
                install_hint="transformers not installed: pip install transformers",
                underlying=underlying,
            )

        assert isinstance(error, EngineDependencyMissingError)
        assert "pip install transformers" in str(error)

    def test_should_classify_top_level_module_not_found_as_missing(self) -> None:
        """The canonical ``ModuleNotFoundError: No module named 'X'`` case."""
        underlying = ModuleNotFoundError("No module named 'faster_whisper'")

        with patch(
            "voicetotext.transcription.backend_imports."
            "importlib.util.find_spec",
            return_value=None,
        ):
            error = make_engine_dependency_error(
                package_name="faster_whisper",
                install_hint="Faster-Whisper not installed: pip install faster-whisper",
                underlying=underlying,
            )

        assert "Faster-Whisper not installed" in str(error)


class TestPackageInstalledButBroken:
    """The user's actual symptom on AMD Windows ROCm: transformers is
    installed but loading it triggers a deep import chain that touches
    ``torch._C._distributed_c10d`` which doesn't exist in AMD's
    Windows torch build.
    """

    def test_should_surface_submodule_failure_when_package_is_findable(self) -> None:
        underlying = ImportError(
            "No module named 'torch._C._distributed_c10d'; "
            "'torch._C' is not a package"
        )

        with patch(
            "voicetotext.transcription.backend_imports."
            "importlib.util.find_spec",
            return_value=MagicMock(),
        ):
            error = make_engine_dependency_error(
                package_name="transformers",
                install_hint="transformers not installed: pip install transformers",
                underlying=underlying,
            )

        message = str(error)
        assert "installed but failed to load" in message
        assert "torch._C._distributed_c10d" in message
        assert "pip install transformers" not in message

    def test_should_name_the_package_in_the_failure_message(self) -> None:
        with patch(
            "voicetotext.transcription.backend_imports."
            "importlib.util.find_spec",
            return_value=MagicMock(),
        ):
            error = make_engine_dependency_error(
                package_name="transformers",
                install_hint="...",
                underlying=ImportError("inner detail"),
            )

        assert "transformers" in str(error)

    def test_should_hint_at_torch_abi_mismatch_in_submodule_failure(self) -> None:
        """The hint must point the user at the real cause (torch ABI)
        because that is the failure mode 99% of the time when a top-
        level package loads but a submodule import dies.
        """
        with patch(
            "voicetotext.transcription.backend_imports."
            "importlib.util.find_spec",
            return_value=MagicMock(),
        ):
            error = make_engine_dependency_error(
                package_name="transformers",
                install_hint="...",
                underlying=ImportError("anything"),
            )

        assert "torch" in str(error).lower()


class TestErrorChaining:
    def test_should_be_an_engine_dependency_missing_error(self) -> None:
        with patch(
            "voicetotext.transcription.backend_imports."
            "importlib.util.find_spec",
            return_value=None,
        ):
            error = make_engine_dependency_error(
                package_name="x",
                install_hint="x not installed",
                underlying=ImportError("y"),
            )

        assert isinstance(error, EngineDependencyMissingError)
