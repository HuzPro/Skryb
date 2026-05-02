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


class _Segment(Protocol):
    text: str


class _CT2WhisperModel(Protocol):
    def transcribe(
        self, audio: object, language: str, beam_size: int,
    ) -> tuple[object, object]: ...


class FasterWhisperAdapter(TranscriptionEngine):
    """Whisper engine backed by faster-whisper / CTranslate2.

    Loads pre-converted CT2 weights from the ``Systran/faster-whisper-*``
    HuggingFace mirror via faster-whisper's own ``download_root`` cache.
    Cross-vendor: the upstream CTranslate2 4.7.1 wheel ships ROCm
    kernels (gfx1100/1101/1151 baked in) and CUDA kernels in the same
    binary, so picking ``device="cuda"`` works on both AMD ROCm and
    NVIDIA installs of PyTorch.
    """

    def __init__(
        self,
        model_name: str,
        repo_id: str,
        device: str,
        language: str,
        beam_size: int,
        model_dir: Path,
    ) -> None:
        self._model_name = model_name
        self._repo_id = repo_id
        self._device = device
        self._language = language
        self._beam_size = beam_size
        self._model_dir = model_dir
        self._model: _CT2WhisperModel | None = None

    def load_model(self) -> None:
        from faster_whisper import WhisperModel

        compute_type = self._resolve_compute_type()
        self._log_phase("load_model", f"start device={self._device} compute_type={compute_type}")

        started_at = time.monotonic()
        self._model = WhisperModel(
            self._repo_id,
            device=self._device,
            compute_type=compute_type,
            download_root=str(self._model_dir),
        )
        self._log_timed_phase("load_model", "model loaded", started_at)

    def _resolve_compute_type(self) -> str:
        if self._device == "cpu":
            return "int8"
        return "float16"

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")

        start = time.monotonic()
        segments, info = self._model.transcribe(
            audio.data,
            language=self._language,
            beam_size=self._beam_size,
        )
        text = "".join(segment.text for segment in segments).strip()

        return TranscriptionResult(
            text=text,
            language=str(getattr(info, "language", self._language)),
            language_probability=float(getattr(info, "language_probability", 0.0)),
            duration_seconds=time.monotonic() - start,
        )

    def dispose(self) -> None:
        if self._model is None:
            return
        self._log_phase("dispose", f"start device={self._device}")
        self._model = None
        self._log_phase("dispose", "refs nulled")
        self._release_gpu_cache_if_needed()
        self._log_phase("dispose", "cache released")

    def _release_gpu_cache_if_needed(self) -> None:
        if self._device == "cpu":
            return
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def _log_phase(self, operation: str, message: str) -> None:
        _logger.info(
            "[%s %s] %s | %s",
            operation, self._model_name, message, vram_snapshot(),
        )

    def _log_timed_phase(
        self, operation: str, message: str, started_at: float,
    ) -> None:
        elapsed = time.monotonic() - started_at
        _logger.info(
            "[%s %s] %s in %.2fs | %s",
            operation, self._model_name, message, elapsed, vram_snapshot(),
        )
