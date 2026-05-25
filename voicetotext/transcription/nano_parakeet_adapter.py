import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    TranscriptionError,
)
from voicetotext.transcription.backend_imports import (
    make_engine_dependency_error,
)
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
        nano_parakeet_module, torch_module = self._import_backend()

        self._log_phase("load_model", f"start device={self._device}")
        started_at = time.monotonic()
        self._load_with_scoped_hf_patch(nano_parakeet_module, torch_module)
        self._log_timed_phase("load_model", "complete", started_at)

    def _import_backend(self) -> tuple[object, object]:
        try:
            import nano_parakeet
            import torch
        except ImportError as exc:
            raise make_engine_dependency_error(
                package_name="nano_parakeet",
                install_hint="Nano-Parakeet not installed: pip install nano-parakeet",
                underlying=exc,
            ) from exc
        return nano_parakeet, torch

    def _load_with_scoped_hf_patch(
        self, nano_parakeet_module: object, torch_module: object,
    ) -> None:
        """Run ``from_pretrained`` with our hf_hub_download patch installed.

        The patch redirects nano-parakeet's internal downloads into the
        app's models dir (see :meth:`_make_patched_download` for why)
        and forces ``local_files_only`` to skip the freshness HEAD
        request. We install it only for the lifetime of the
        ``from_pretrained`` call and restore the original in a
        ``finally`` block — without that, back-to-back adapter loads
        stomp each other's closures and the module-global stays
        tainted long after the engine has been disposed.

        Force float32 because nano-parakeet auto-selects bfloat16 on
        bf16-capable GPUs (RDNA4 included) and ROCm's bf16 kernels
        are less battle-tested than fp32 — the perf/memory cost is
        worth the stability.
        """
        original_download = nano_parakeet_module.hf_hub_download
        nano_parakeet_module.hf_hub_download = self._make_patched_download(
            original_download,
        )
        try:
            self._model = nano_parakeet_module.from_pretrained(
                model_name=self._repo_id,
                device=self._device,
                dtype=torch_module.float32,
            )
        finally:
            nano_parakeet_module.hf_hub_download = original_download

    def _make_patched_download(
        self, original: Callable[..., object],
    ) -> Callable[..., object]:
        """Build the closure that redirects calls into ``model_dir``.

        nano-parakeet calls ``huggingface_hub.hf_hub_download(repo_id,
        filename)`` with no ``cache_dir``, so it falls through to the
        global default (``~/.cache/huggingface/hub``). Our chunked
        downloader writes to ``self._model_dir`` — so without the
        redirect an already-downloaded model is invisible. The save
        guard in ``QtSettingsWindow._on_save_clicked`` has already
        verified the file is on disk, so ``local_files_only=True``
        short-circuits the upstream HEAD-request round trip.
        """
        cache_dir = str(self._model_dir)

        def patched(*args: object, **kwargs: object) -> object:
            kwargs.setdefault("cache_dir", cache_dir)
            kwargs.setdefault("local_files_only", True)
            return original(*args, **kwargs)

        return patched

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
            transcription_seconds=time.monotonic() - start,
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
