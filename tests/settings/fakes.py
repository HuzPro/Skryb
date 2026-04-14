"""Backwards-compat shim.

The canonical home for these fakes is :mod:`tests.fakes`.  This module
re-exports them so existing imports keep working during the transition.
New tests should import from ``tests.fakes`` directly.
"""

from tests.fakes.config import FakeConfig
from tests.fakes.model_manager import FakeModelManager

__all__ = ["FakeConfig", "FakeModelManager"]
