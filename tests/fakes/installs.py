"""What an install ships, for tests whose answer depends on it."""

import importlib.util
from contextlib import contextmanager
from collections.abc import Iterator
from unittest.mock import patch

from skryb.transcription.engine_factory import build_engine_registry
from skryb.transcription.engine_registry import EngineRegistry


@contextmanager
def without_openvino() -> Iterator[None]:
    """An AMD or NVIDIA install: only the CPU build ships OpenVINO, whose NPU engines make the ranking wait for
    the hardware probe."""
    find_spec = importlib.util.find_spec
    with patch.object(importlib.util, "find_spec", lambda name, *args: None if name == "openvino" else find_spec(name, *args)):
        yield


def gpu_install_registry() -> EngineRegistry:
    with without_openvino():
        return build_engine_registry()
