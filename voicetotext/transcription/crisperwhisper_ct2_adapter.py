"""CrisperWhisper 2.0 inference through the CTranslate2 runtime."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.backend_imports import (
    make_engine_dependency_error,
)
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_types import CrisperWhisperMode
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.whisper_features import (
    log_mel_spectrogram,
    split_into_encoder_windows,
)
from voicetotext.transcription.whisper_tokens import WhisperTokens


if TYPE_CHECKING:
    from ctranslate2.models import Whisper


_logger = logging.getLogger(__name__)

_CPU_DEVICE = "cpu"
# CTranslate2 halves whatever budget it is given
# (min(max_length / 2, max_length - start_step)), so the doubled value is
# what yields the 438 new tokens the model's 448 decoder positions allow
# after the nine-token prompt.
_MAX_LENGTH = 876
# bfloat16 matches the checkpoint on disk, so no weights are rewritten at
# load. CPU has no efficient bfloat16, and asking for it there makes
# CTranslate2 convert to float32 while logging a warning; "default" lets
# it pick that itself, quietly.
_GPU_COMPUTE_TYPE = "bfloat16"
_CPU_COMPUTE_TYPE = "default"


class _WhisperRuntime(Protocol):
    def generate(
        self,
        features: object,
        prompts: list[list[int]],
        *,
        beam_size: int,
        sampling_topk: int,
        suppress_blank: bool,
        max_length: int,
    ) -> list[_WhisperResult]: ...


class _WhisperResult(Protocol):
    @property
    def sequences_ids(self) -> list[list[int]]: ...


class CrisperWhisperCt2Adapter(TranscriptionEngine):
    """Whisper adapter that never imports torch or Transformers.

    Decodes with the same explicit prompt as the Transformers adapter:
    five atomic mode tags followed by Whisper's own control tokens.
    CTranslate2 locates ``<|startoftranscript|>`` wherever it sits in the
    prompt, so the leading mode tags reach the decoder as context rather
    than being mistaken for the start of the sequence.

    *model_dir* is the converted CTranslate2 directory built from the
    pinned snapshot, so the weights are the audited revision's.
    """

    def __init__(
        self,
        model_dir: Path,
        device: str,
        language: str,
        mode: CrisperWhisperMode = "intended",
    ) -> None:
        self._model_dir = model_dir
        self._device = device
        self._language = language
        self._mode = mode
        self._model: _WhisperRuntime | None = None
        self._tokens: WhisperTokens | None = None
        self._prompt: list[int] | None = None

    def load_model(self) -> None:
        started_at = time.monotonic()
        whisper_type = _import_backend()
        tokens = WhisperTokens(self._model_dir)
        model = whisper_type(
            str(self._model_dir),
            device=self._device,
            compute_type=self._compute_type(),
        )
        self._tokens = tokens
        self._model = model
        self._prompt = tokens.decoder_prompt(self._language, self._mode)
        _logger.info(
            "[load_model %s] complete in %.2fs",
            self._model_dir.name,
            time.monotonic() - started_at,
        )

    def is_model_loaded(self) -> bool:
        return self._model is not None and self._tokens is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        started_at = time.monotonic()
        text = self._transcribe_every_window(audio)
        return TranscriptionResult(
            text=text,
            language=self._language,
            language_probability=0.0,
            transcription_seconds=time.monotonic() - started_at,
        )

    def _transcribe_every_window(self, audio: AudioSegment) -> str:
        parts = [
            self._generate_text(window).strip()
            for window in split_into_encoder_windows(audio)
        ]
        return " ".join(part for part in parts if part)

    def _generate_text(self, audio: AudioSegment) -> str:
        model, tokens, prompt = self._loaded()
        features = np.ascontiguousarray(
            log_mel_spectrogram(audio)[np.newaxis, :, :],
        )
        results = model.generate(
            _storage_view(features),
            [prompt],
            beam_size=1,
            sampling_topk=1,
            suppress_blank=False,
            max_length=_MAX_LENGTH,
        )
        return tokens.decode(results[0].sequences_ids[0])

    def _loaded(self) -> tuple[_WhisperRuntime, WhisperTokens, list[int]]:
        model, tokens, prompt = self._model, self._tokens, self._prompt
        if model is None or tokens is None or prompt is None:
            raise TranscriptionError("Model not loaded")
        return model, tokens, prompt

    def _compute_type(self) -> str:
        if self._device == _CPU_DEVICE:
            return _CPU_COMPUTE_TYPE
        return _GPU_COMPUTE_TYPE

    def dispose(self) -> None:
        self._model = None
        self._tokens = None
        self._prompt = None


def _storage_view(features: np.ndarray) -> object:
    import ctranslate2

    return ctranslate2.StorageView.from_array(features)


def _import_backend() -> type[Whisper]:
    try:
        from ctranslate2.models import Whisper
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="ctranslate2",
            install_hint=(
                "CTranslate2 not installed: see INSTALL.md for the "
                "ROCm wheel this engine needs"
            ),
            underlying=exc,
        ) from exc
    return Whisper
