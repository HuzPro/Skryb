import logging
import time
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

_WHISPER_DECODING_DEFAULTS: dict[str, object] = {
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 1.35,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
}


class _WhisperModel(Protocol):
    dtype: object

    def to(self, device: str) -> "_WhisperModel": ...
    def half(self) -> "_WhisperModel": ...
    def generate(self, input_features: object, **kwargs: object) -> object: ...


class _WhisperProcessor(Protocol):
    def __call__(self, audio: object, **kwargs: object) -> object: ...
    def batch_decode(self, token_ids: object, skip_special_tokens: bool = ...) -> list[str]: ...


class TransformersWhisperAdapter(TranscriptionEngine):
    """Whisper engine backed by HuggingFace *transformers* (pure PyTorch).

    Works on any GPU that PyTorch supports, including AMD via ROCm.
    Models are downloaded from HuggingFace Hub (``openai/whisper-*``).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str = "en",
        beam_size: int = 5,
        download_root: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._language = language
        self._beam_size = beam_size
        self._download_root = download_root
        self._model: _WhisperModel | None = None
        self._processor: _WhisperProcessor | None = None

    def load_model(self) -> None:
        whisper_model_class, whisper_processor_class = self._import_backend()

        self._log_phase("load_model", f"start device={self._device}")

        repo_id = f"openai/whisper-{self._model_size}"
        shared_kwargs: dict[str, object] = {}
        if self._download_root:
            shared_kwargs["cache_dir"] = self._download_root
            shared_kwargs["local_files_only"] = True

        started_at = time.monotonic()
        self._processor = whisper_processor_class.from_pretrained(
            repo_id, **shared_kwargs,
        )
        self._log_timed_phase("load_model", "processor loaded", started_at)

        started_at = time.monotonic()
        self._model = whisper_model_class.from_pretrained(
            repo_id, **self._model_load_kwargs(shared_kwargs)
        )
        self._log_timed_phase("load_model", "model loaded", started_at)

        self._optimize_for_device()

        self._log_phase("load_model", "complete")

    def _import_backend(self) -> tuple[object, object]:
        try:
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )
        except ImportError as exc:
            raise make_engine_dependency_error(
                package_name="transformers",
                install_hint="transformers not installed: pip install transformers",
                underlying=exc,
            ) from exc
        return WhisperForConditionalGeneration, WhisperProcessor

    def _model_load_kwargs(self, base: dict[str, object]) -> dict[str, object]:
        # HF 5.x honors config.json ``torch_dtype`` (fp16 for large-v3-turbo),
        # which breaks CPU Linear/Conv layers - force fp32 on CPU.
        # Explicit ``attn_implementation="sdpa"`` opts into the SDPA
        # path (Flash / mem-efficient on GPU, scaled-dot-product on
        # CPU) rather than relying on transformers' default - 20–30%
        # faster on every backend we ship.
        kwargs = {**base, "attn_implementation": "sdpa"}
        if self._device != "cpu":
            return kwargs
        import torch
        return {**kwargs, "dtype": torch.float32}

    def _optimize_for_device(self) -> None:
        if self._device == "cpu":
            return
        self._move_model_to_device()
        self._cast_to_half_precision_if_cuda_available()

    def _move_model_to_device(self) -> None:
        started_at = time.monotonic()
        self._model = self._model.to(self._device)
        self._log_timed_phase("load_model", f"model.to({self._device})", started_at)

    def _cast_to_half_precision_if_cuda_available(self) -> None:
        import torch
        if not torch.cuda.is_available():
            return
        started_at = time.monotonic()
        self._model = self._model.half()
        self._log_timed_phase("load_model", "model.half()", started_at)

    def is_model_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")

        import torch

        start = time.monotonic()
        input_features, attention_mask = self._preprocess_audio(audio)
        generate_kwargs = self._build_generate_kwargs(attention_mask)

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features, **generate_kwargs
            )

        text = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return TranscriptionResult(
            text=text,
            language=self._language,
            language_probability=0.0,
            transcription_seconds=time.monotonic() - start,
        )

    def _preprocess_audio(self, audio: AudioSegment) -> tuple[object, object]:
        processed = self._processor(
            audio.data,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
        )
        input_features = processed.input_features.to(self._device)
        attention_mask = processed.attention_mask.to(self._device)

        if self._device != "cpu" and self._model.dtype != input_features.dtype:
            input_features = input_features.half()

        return input_features, attention_mask

    def _build_generate_kwargs(self, attention_mask: object) -> dict[str, object]:
        return {
            "num_beams": self._beam_size,
            "return_timestamps": True,
            "attention_mask": attention_mask,
            "condition_on_prev_tokens": True,
            "language": self._language,
            "task": "transcribe",
            **_WHISPER_DECODING_DEFAULTS,
        }

    def dispose(self) -> None:
        # Skip vram_snapshot on un-loaded adapters so tests that never
        # call load_model don't force a real torch import.
        if self._model is None and self._processor is None:
            return
        self._log_phase("dispose", f"start device={self._device}")
        self._model = None
        self._processor = None
        self._log_phase("dispose", "refs nulled")
        if self._device != "cpu":
            import torch
            torch.cuda.empty_cache()
        self._log_phase("dispose", "cache released")

    def _log_phase(self, operation: str, message: str) -> None:
        _logger.info(
            "[%s %s] %s | %s",
            operation, self._model_size, message, vram_snapshot(),
        )

    def _log_timed_phase(
        self, operation: str, message: str, started_at: float,
    ) -> None:
        elapsed = time.monotonic() - started_at
        _logger.info(
            "[%s %s] %s in %.2fs | %s",
            operation, self._model_size, message, elapsed, vram_snapshot(),
        )
