import logging
import time
from pathlib import Path
from typing import Protocol

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_diagnostics import vram_snapshot
from voicetotext.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)


class _ParakeetModel(Protocol):
    def transcribe(self, audio: object) -> str: ...


class NanoParakeetAdapter(TranscriptionEngine):
    """Parakeet TDT engine backed by *nano-parakeet* (pure PyTorch).

    Works on any GPU that PyTorch supports, including AMD via ROCm.
    Models are downloaded from HuggingFace Hub (``nvidia/parakeet-tdt-*``).
    """

    def __init__(
        self,
        repo_id: str,
        device: str,
        model_dir: Path,
    ) -> None:
        self._repo_id = repo_id
        self._device = device
        self._model_dir = model_dir
        self._model: _ParakeetModel | None = None

    def load_model(self) -> None:
        import nano_parakeet
        import torch

        self._log_phase("load_model", f"start device={self._device}")
        started_at = time.monotonic()

        self._patch_hf_hub_download(nano_parakeet)
        # Force float32: nano_parakeet auto-selects bfloat16 on bf16-
        # capable GPUs (RDNA4 included), and ROCm's bf16 kernels are
        # less battle-tested than fp32. The slight memory/perf cost is
        # worth the stability.
        self._model = nano_parakeet.from_pretrained(
            model_name=self._repo_id,
            device=self._device,
            dtype=torch.float32,
        )

        self._log_timed_phase("load_model", "complete", started_at)

    def _patch_hf_hub_download(self, nano_parakeet_module: object) -> None:
        """Force nano-parakeet to look in our app's models dir, offline.

        Two reasons:

        1. nano-parakeet calls ``huggingface_hub.hf_hub_download(repo_id,
           filename)`` with no ``cache_dir``, so it falls through to the
           global default (``~/.cache/huggingface/hub``). Our chunked
           downloader writes to ``self._model_dir`` — so without
           ``cache_dir`` an already-downloaded model is invisible.

        2. By the time the adapter loads, the save guard in
           ``QtSettingsWindow._on_save_clicked`` has already verified
           ``is_installed(model_name)`` is True. The file IS on disk.
           Default ``hf_hub_download`` still does a HEAD request to the
           Hub to validate freshness — slow, leaks "unauthenticated
           request" warnings, and fails on a flaky network. We pass
           ``local_files_only=True`` to short-circuit that round trip.
        """
        original = nano_parakeet_module.hf_hub_download
        cache_dir = str(self._model_dir)

        def patched(*args: object, **kwargs: object) -> object:
            kwargs.setdefault("cache_dir", cache_dir)
            kwargs.setdefault("local_files_only", True)
            return original(*args, **kwargs)

        nano_parakeet_module.hf_hub_download = patched

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")

        start = time.monotonic()
        text = self._model.transcribe(audio.data)

        return TranscriptionResult(
            text=str(text).strip(),
            language="en",
            language_probability=0.0,
            duration_seconds=time.monotonic() - start,
        )

    def dispose(self) -> None:
        if self._model is None:
            return
        self._log_phase("dispose", f"start device={self._device}")
        self._model = None
        self._log_phase("dispose", "refs nulled")
        if self._device != "cpu":
            import torch
            torch.cuda.empty_cache()
        self._log_phase("dispose", "cache released")

    def _log_phase(self, operation: str, message: str) -> None:
        _logger.info(
            "[%s %s] %s | %s",
            operation, self._repo_id, message, vram_snapshot(),
        )

    def _log_timed_phase(
        self, operation: str, message: str, started_at: float,
    ) -> None:
        elapsed = time.monotonic() - started_at
        _logger.info(
            "[%s %s] %s in %.2fs | %s",
            operation, self._repo_id, message, elapsed, vram_snapshot(),
        )
