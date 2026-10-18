import logging
import time
from pathlib import Path
from typing import Protocol

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_diagnostics import vram_snapshot
from voicetotext.transcription.parakeet_loader import ParakeetModelLoader
from voicetotext.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)


# Long-audio inference allocates encoder activations proportional to the
# clip length. On the 9-minute LongStory fixture the unchunked path peaks
# at ~9 GB of VRAM, barely fitting on a 16 GB card and OOMing on smaller
# ones. Splitting into 60 s chunks caps the peak under ~4.2 GB with
# negligible extra latency (see
# ``tests/performance/.results/dtype_comparison_chunked_results.json``).
_CHUNK_SECONDS = 60.0


class _ParakeetModel(Protocol):
    def transcribe(self, audio: object) -> str: ...


class NanoParakeetAdapter(TranscriptionEngine):
    """Parakeet TDT engine backed by *nano-parakeet* (pure PyTorch).

    Works on any GPU that PyTorch supports, including AMD via ROCm.
    Models are downloaded from HuggingFace Hub (``nvidia/parakeet-tdt-*``,
    ``moondream/parakeet-ultra``).
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
        self._log_phase("load_model", f"start device={self._device}")
        started_at = time.monotonic()
        self._model = ParakeetModelLoader(
            self._repo_id, self._model_dir,
        ).load(self._device)
        self._log_timed_phase("load_model", "complete", started_at)

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        self._raise_if_not_loaded()
        start = time.monotonic()
        text = self._transcribe_chunked(audio)
        return TranscriptionResult(
            text=text,
            language="en",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - start,
        )

    def _raise_if_not_loaded(self) -> None:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")

    def _transcribe_chunked(self, audio: AudioSegment) -> str:
        chunks = self._split_into_chunks(audio)
        parts = [
            str(self._model.transcribe(chunk.data)).strip()
            for chunk in chunks
        ]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _split_into_chunks(audio: AudioSegment) -> list[AudioSegment]:
        samples_per_chunk = int(_CHUNK_SECONDS * audio.sample_rate)
        total_samples = audio.sample_count
        chunks: list[AudioSegment] = []
        for start in range(0, total_samples, samples_per_chunk):
            end = min(start + samples_per_chunk, total_samples)
            chunk_data = audio.data[start:end]
            if chunk_data.size == 0:
                continue
            chunks.append(
                AudioSegment(data=chunk_data, sample_rate=audio.sample_rate),
            )
        return chunks

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
