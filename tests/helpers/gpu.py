"""Skip helper for tests that require a ROCm/CUDA GPU.

Tests that load a real speech model on the GPU call :func:`require_cuda`
before doing any work; if PyTorch is missing or
``torch.cuda.is_available()`` returns ``False`` the test is skipped
with a clear reason rather than failing.
"""

import pytest


def require_cuda() -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm GPU not available")
