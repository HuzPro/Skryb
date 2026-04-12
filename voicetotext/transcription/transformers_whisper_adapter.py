import logging
import time
from typing import Protocol

from voicetotext.transcription.engine_diagnostics import vram_snapshot
from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)

_TEMPERATURE_FALLBACK_SCHEDULE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
_COMPRESSION_RATIO_THRESHOLD = 1.35
_LOGPROB_THRESHOLD = -1.0
_NO_SPEECH_THRESHOLD = 0.6


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
        language: str | None = None,
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
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        _logger.info(
            "[load_model %s device=%s] start | %s",
            self._model_size,
            self._device,
            vram_snapshot(),
        )

        repo_id = f"openai/whisper-{self._model_size}"
        kwargs = {}
        if self._download_root:
            kwargs["cache_dir"] = self._download_root
            kwargs["local_files_only"] = True

        t0 = time.monotonic()
        self._processor = WhisperProcessor.from_pretrained(repo_id, **kwargs)
        _logger.info(
            "[load_model %s] processor loaded in %.2fs",
            self._model_size,
            time.monotonic() - t0,
        )

        t1 = time.monotonic()
        self._model = WhisperForConditionalGeneration.from_pretrained(repo_id, **kwargs)
        _logger.info(
            "[load_model %s] model loaded in %.2fs | %s",
            self._model_size,
            time.monotonic() - t1,
            vram_snapshot(),
        )

        self._optimize_for_device()

        _logger.info(
            "[load_model %s] complete | %s", self._model_size, vram_snapshot()
        )

    def _optimize_for_device(self) -> None:
        if self._device == "cpu":
            return

        import torch

        t0 = time.monotonic()
        self._model = self._model.to(self._device)
        _logger.info(
            "[load_model %s] model.to(%s) in %.2fs | %s",
            self._model_size,
            self._device,
            time.monotonic() - t0,
            vram_snapshot(),
        )

        if not torch.cuda.is_available():
            return

        t1 = time.monotonic()
        self._model = self._model.half()
        _logger.info(
            "[load_model %s] model.half() in %.2fs | %s",
            self._model_size,
            time.monotonic() - t1,
            vram_snapshot(),
        )

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

        language = self._language or "en"
        return TranscriptionResult(
            text=text,
            language=language,
            language_probability=0.0,
            duration_seconds=time.monotonic() - start,
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
        kwargs: dict[str, object] = {
            "num_beams": self._beam_size,
            "return_timestamps": True,
            "attention_mask": attention_mask,
            "temperature": _TEMPERATURE_FALLBACK_SCHEDULE,
            "compression_ratio_threshold": _COMPRESSION_RATIO_THRESHOLD,
            "logprob_threshold": _LOGPROB_THRESHOLD,
            "no_speech_threshold": _NO_SPEECH_THRESHOLD,
            "condition_on_prev_tokens": True,
        }
        if self._language:
            kwargs["language"] = self._language
            kwargs["task"] = "transcribe"
        return kwargs

    def dispose(self) -> None:
        # Early-return on un-loaded adapters so tests that construct
        # an adapter but never call ``load_model`` don't force a real
        # ``import torch`` through ``vram_snapshot``.
        if self._model is None and self._processor is None:
            return
        _logger.info(
            "[dispose %s device=%s] start | %s",
            self._model_size,
            self._device,
            vram_snapshot(),
        )
        self._model = None
        self._processor = None
        _logger.info(
            "[dispose %s] refs nulled | %s",
            self._model_size,
            vram_snapshot(),
        )
        if self._device != "cpu":
            import torch
            torch.cuda.empty_cache()
        _logger.info(
            "[dispose %s] cache released | %s",
            self._model_size,
            vram_snapshot(),
        )
