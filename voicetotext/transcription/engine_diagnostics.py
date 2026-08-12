"""Runtime diagnostics for the transcription engine.

Provides VRAM and adapter-instance snapshots for the model hot-swap
path. These live in the transcription layer (not core) because they
depend on torch and the engine ABC.
"""

import gc
import sys


def vram_snapshot() -> str:
    """Return a short ``allocated/reserved`` VRAM string, or a sentinel.

    Pure read - never raises, never blocks. Returns
    ``torch-not-imported`` if torch has not yet been loaded by the
    process, so calling this from a test that mocks torch in
    ``sys.modules`` never triggers a slow real import.
    """
    if "torch" not in sys.modules:
        return "torch-not-imported"
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda-unavailable"
        mb = 1024 * 1024
        allocated = torch.cuda.memory_allocated() / mb
        reserved = torch.cuda.memory_reserved() / mb
        return f"allocated={allocated:.0f}MB reserved={reserved:.0f}MB"
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"snapshot-failed({exc})"


def count_live_adapters() -> int:
    """Count live :class:`TranscriptionEngine` instances on the heap.

    Used by the hot-swap coordinator to verify the previous adapter was
    actually released after a swap. Returns ``-1`` on any failure and
    never raises into the caller. Counts every ``TranscriptionEngine``
    subclass - adding a new engine does not require updating this fn.
    """
    try:
        from voicetotext.transcription.engine import TranscriptionEngine

        return sum(
            1 for obj in gc.get_objects() if isinstance(obj, TranscriptionEngine)
        )
    except Exception:  # pragma: no cover - diagnostic only
        return -1
