import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.backend_imports import (
    make_engine_dependency_error,
)
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_diagnostics import vram_snapshot
from voicetotext.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)


_CPU_DEVICE = "cpu"


class _CtcInputs(Protocol):
    def to(self, device: str, *, dtype: object) -> Mapping[str, object]: ...


class _CtcModel(Protocol):
    @property
    def dtype(self) -> object: ...
    def generate(self, **features: object) -> object: ...


class _CtcProcessor(Protocol):
    def __call__(
        self, audio: object, *, sampling_rate: int, device: str,
    ) -> _CtcInputs: ...
    def batch_decode(
        self, sequences: object, *, skip_special_tokens: bool,
    ) -> list[str]: ...


class GraniteTurboCtcAdapter(TranscriptionEngine):
    """Granite Speech 5.0 TurboCTC engine (transformers, English-only).

    Encoder-only Conformer with self-conditioned CTC: transcription is a
    single non-autoregressive forward pass plus a greedy argmax
    (``GraniteSpeech5ForCTC.generate``), making this the lowest-latency
    batch engine in the app. Weights ship in bf16 and load via the
    standard HuggingFace cache layout the app's downloader writes.
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
        self._model: _CtcModel | None = None
        self._processor: _CtcProcessor | None = None

    def load_model(self) -> None:
        self._log_phase("load_model", f"start device={self._device}")
        started_at = time.monotonic()
        model_class, processor_class, torch_module = self._import_backend()
        self._processor = processor_class.from_pretrained(
            self._repo_id,
            cache_dir=str(self._model_dir),
            local_files_only=True,
        )
        model = model_class.from_pretrained(
            self._repo_id,
            cache_dir=str(self._model_dir),
            local_files_only=True,
            dtype=self._dtype_for(torch_module),
            attn_implementation="sdpa",
        )
        self._model = model.to(self._device)
        self._log_timed_phase("load_model", "complete", started_at)

    def _import_backend(self) -> tuple[object, object, object]:
        try:
            import torch
            from transformers import (
                GraniteSpeech5ForCTC,
                GraniteSpeech5Processor,
            )
        except ImportError as exc:
            raise make_engine_dependency_error(
                package_name="transformers",
                install_hint=(
                    "transformers not installed: "
                    "pip install 'transformers>=5.16.0,<6' 'torchaudio>=2.5'"
                ),
                underlying=exc,
            ) from exc
        return GraniteSpeech5ForCTC, GraniteSpeech5Processor, torch

    def _dtype_for(self, torch_module: object) -> object:
        """bf16 on GPU, fp32 on CPU.

        The checkpoint ships bf16 weights and the conv batch norms stay
        fp32 via ``_keep_in_fp32_modules_strict``, so bf16 is the fast
        path on ROCm/CUDA. CPU has no fast bf16 matmul, so fp32 is both
        correct and faster there (same policy as the Parakeet loader).
        """
        if self._device == _CPU_DEVICE:
            return torch_module.float32
        return torch_module.bfloat16

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        self._raise_if_not_loaded()
        start = time.monotonic()
        text = self._decode_greedy(audio)
        return TranscriptionResult(
            text=text,
            language="en",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - start,
        )

    def _raise_if_not_loaded(self) -> None:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")

    def _decode_greedy(self, audio: AudioSegment) -> str:
        features = self._processor(
            audio.data,
            sampling_rate=audio.sample_rate,
            device=self._device,
        ).to(self._device, dtype=self._model.dtype)
        sequences = self._model.generate(**features)
        texts = self._processor.batch_decode(
            sequences, skip_special_tokens=True,
        )
        return str(texts[0]).strip()

    def dispose(self) -> None:
        if self._model is None:
            return
        self._log_phase("dispose", f"start device={self._device}")
        self._model = None
        self._processor = None
        self._log_phase("dispose", "refs nulled")
        if self._device != _CPU_DEVICE:
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
