"""Qwen3-ASR through transformers (PyTorch): the default engine wherever the native one cannot run."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.backend_imports import make_engine_dependency_error
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.engine_diagnostics import vram_snapshot
from skryb.transcription.qwen3_asr_prompt import LANGUAGE_NAMES
from skryb.transcription.result import TranscriptionResult
from skryb.transcription.roman_urdu import romanize

_logger = logging.getLogger(__name__)

_CPU_DEVICE = "cpu"
# The model card's bound per request; a meeting clip is cut at 15 s, well inside it.
_MAX_NEW_TOKENS = 256


class Qwen3AsrAdapter(TranscriptionEngine):
    """Loads a ``Qwen/Qwen3-ASR-*-hf`` snapshot (plain transformers weights, no remote code).

    bf16 on a GPU, the checkpoint's own dtype; fp32 on the CPU, which has no fast bf16.
    """

    def __init__(self, snapshot: Path, device: str, language: str) -> None:
        self._snapshot = snapshot
        self._device = device
        self._language = language if language in LANGUAGE_NAMES else "en"
        self._processor: Any = None
        self._model: Any = None
        self._dtype: Any = None

    def load_model(self) -> None:
        started_at = time.monotonic()
        torch, transformers = _import_backend()
        self._dtype = torch.float32 if self._device == _CPU_DEVICE else torch.bfloat16
        self._processor = transformers.AutoProcessor.from_pretrained(self._snapshot)
        model = transformers.Qwen3ASRForConditionalGeneration.from_pretrained(
            self._snapshot, dtype=self._dtype, attn_implementation="sdpa",
        )
        self._model = model.to(self._device).eval()
        _logger.info(
            "[load_model qwen3-asr] complete on %s in %.2fs | %s",
            self._device, time.monotonic() - started_at, vram_snapshot(),
        )

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        return self.transcribe_prompted(audio, None, self._language)

    def transcribe_prompted(
        self, audio: AudioSegment, context: str | None, language: str | None,
    ) -> TranscriptionResult:
        if self._model is None:
            raise TranscriptionError("Model not loaded")
        import torch

        started_at = time.monotonic()
        inputs = self._processor.apply_transcription_request(
            audio=np.ascontiguousarray(audio.data, dtype=np.float32), language=language, prompt=context,
        ).to(self._device, self._dtype)
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
        text = self._processor.decode(output[0, prompt_length:], return_format="transcription_only")
        return TranscriptionResult(
            # Qwen3-ASR has no Urdu: it writes spoken Urdu in Hindi script.
            text=romanize(str(text).strip()),
            language=language or "",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - started_at,
        )

    def dispose(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._processor = None
        if self._device != _CPU_DEVICE:
            import torch

            torch.cuda.empty_cache()
        _logger.info("[dispose qwen3-asr] released | %s", vram_snapshot())


def _import_backend() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="transformers",
            install_hint='transformers not installed: pip install "transformers>=5.18.0,<6"',
            underlying=exc,
        ) from exc
    return torch, transformers
