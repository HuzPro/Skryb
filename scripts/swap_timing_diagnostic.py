"""Diagnostic script: multi-swap timing comparison.

Loads models in sequence: large-v3 -> turbo -> small -> large-v3 -> turbo
Prints per-step timing so we can compare first vs second load of each model
and pinpoint where the 39-second delay lives.

Run from project root:
    python scripts/swap_timing_diagnostic.py
"""
import gc
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-5s] %(message)s",
)
_logger = logging.getLogger(__name__)


def _vram():
    import torch
    mb = 1024 * 1024
    a = torch.cuda.memory_allocated() / mb
    r = torch.cuda.memory_reserved() / mb
    return f"alloc={a:.0f}MB res={r:.0f}MB"


def load(model_size: str, device: str, download_root: str):
    from voicetotext.transcription.transformers_whisper_adapter import (
        TransformersWhisperAdapter,
    )
    adapter = TransformersWhisperAdapter(
        model_size=model_size, device=device, download_root=download_root,
    )
    t0 = time.monotonic()
    adapter.load_model()
    elapsed = time.monotonic() - t0
    _logger.info("LOAD  %-16s  %.2fs  | %s", model_size, elapsed, _vram())
    return adapter


def dispose(adapter):
    name = adapter._model_size
    _logger.info("DISPOSE %-14s start | %s", name, _vram())

    t0 = time.monotonic()
    adapter._model = None
    adapter._processor = None
    t1 = time.monotonic()
    _logger.info("  refs nulled       %.2fs | %s", t1 - t0, _vram())

    gc.collect()
    t2 = time.monotonic()
    _logger.info("  gc.collect        %.2fs | %s", t2 - t1, _vram())

    import torch
    torch.cuda.empty_cache()
    t3 = time.monotonic()
    _logger.info("  empty_cache       %.2fs | %s", t3 - t2, _vram())

    _logger.info("DISPOSE %-14s total %.2fs | %s", name, t3 - t0, _vram())


def main():
    import os
    download_root = os.path.join(
        os.environ["APPDATA"], "VoiceToText", "models",
    )
    device = "cuda"

    sequence = ["large-v3", "large-v3-turbo", "small", "large-v3", "large-v3-turbo"]

    _logger.info("=== swap timing diagnostic ===")
    _logger.info("sequence: %s", " -> ".join(sequence))
    _logger.info("")

    adapter = load(sequence[0], device, download_root)

    for next_model in sequence[1:]:
        _logger.info("--- swap to %s ---", next_model)
        dispose(adapter)
        _logger.info("")
        adapter = load(next_model, device, download_root)
        _logger.info("")

    dispose(adapter)
    _logger.info("=== done ===")


if __name__ == "__main__":
    main()
