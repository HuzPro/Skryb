"""Import torch and warm GPU kernels before the engine needs them."""

from __future__ import annotations

import importlib
import logging
import threading
from concurrent.futures import Future
from types import ModuleType


_logger = logging.getLogger(__name__)

# Representative decode-path shapes. The first call into a kernel family
# costs driver setup rather than compute, so warming with approximate
# shapes recovers all but ~0.04s of what the model's exact shapes would.
_LOGIT_WIDTH = 51200
_FEATURE_WIDTH = 1280
_TOKEN_SAMPLE = 199
_MEL_BINS = 128
_ENCODER_FRAMES = 3000
_CONV_KERNEL = 3


def start_torch_prewarm() -> Future[ModuleType]:
    """Import torch on a background thread, then warm the GPU on it.

    The future resolves as soon as torch is imported, so startup is never
    blocked by the warming that continues behind it.
    """
    future: Future[ModuleType] = Future()

    threading.Thread(
        target=lambda: _import_then_warm(future),
        daemon=True,
        name="torch-prewarm",
    ).start()
    return future


def _import_then_warm(future: Future[ModuleType]) -> None:
    try:
        torch = importlib.import_module("torch")
    except BaseException as exc:
        future.set_exception(exc)
        return
    future.set_result(torch)
    _warm_gpu_kernels(torch)


def _warm_gpu_kernels(torch: ModuleType) -> None:
    """Pay the per-process GPU first-call cost off the critical path.

    Every launch spends seconds of CPU-bound driver setup the first time
    it touches each kernel family, and no on-disk cache covers it. Doing
    it here hides it behind the backend import and model load that
    follow. Warming is best effort: failing to warm must never stop the
    app from starting.
    """
    try:
        _touch_decode_kernels(torch)
    except Exception:
        _logger.debug("GPU prewarm skipped", exc_info=True)


def _touch_decode_kernels(torch: ModuleType) -> None:
    if not torch.cuda.is_available():
        return
    device = "cuda"
    dtype = torch.bfloat16
    functional = torch.nn.functional

    torch.empty(1, device=device)

    features = torch.randn(
        _TOKEN_SAMPLE, _FEATURE_WIDTH, dtype=dtype, device=device,
    )
    projection = torch.randn(
        _FEATURE_WIDTH, _FEATURE_WIDTH, dtype=dtype, device=device,
    )
    features @ projection
    functional.gelu(features)
    functional.layer_norm(features.float(), (_FEATURE_WIDTH,))

    vocab = torch.arange(_LOGIT_WIDTH, device=device)
    tokens = torch.arange(1, _TOKEN_SAMPLE + 1, device=device)
    vocab.sort()
    vocab.cumsum(0)
    torch.isin(vocab, tokens)
    torch.searchsorted(vocab, tokens)

    logits = torch.randn(1, _LOGIT_WIDTH, device=device)
    torch.where(logits > 0, logits, logits)
    logits.softmax(-1)
    logits.argmax(-1)

    functional.conv1d(
        torch.randn(
            1, _MEL_BINS, _ENCODER_FRAMES, dtype=dtype, device=device,
        ),
        torch.randn(
            _FEATURE_WIDTH, _MEL_BINS, _CONV_KERNEL, dtype=dtype,
            device=device,
        ),
    )
    torch.cuda.synchronize()
