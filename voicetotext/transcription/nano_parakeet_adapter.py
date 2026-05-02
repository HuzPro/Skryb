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
        from nano_parakeet import from_pretrained

        self._log_phase("load_model", f"start device={self._device}")
        started_at = time.monotonic()

        self._model = from_pretrained(
            model_name=self._repo_id,
            device=self._device,
        )

        self._log_timed_phase("load_model", "complete", started_at)

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
