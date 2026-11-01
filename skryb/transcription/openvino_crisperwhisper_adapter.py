"""CrisperWhisper 2.0 with its encoder on an Intel NPU and its decoder on the CPU, through OpenVINO, no torch.

Decodes as the native GPU engine does: Nyra's intended/verbatim prompt from the model's own tokenizer,
greedy, the generation config's suppressed tokens, no first-token suppression, and long audio in 30 s
windows that carry the words already heard (``transcribe_in_windows``). The heavy lifting is
``openvino_whisper``.
"""

import logging
import time
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.engine_types import CrisperWhisperMode
from skryb.transcription.openvino_whisper import OpenVinoWhisper
from skryb.transcription.result import TranscriptionResult
from skryb.transcription.whisper_features import transcribe_in_windows
from skryb.transcription.whisper_tokens import WhisperTokens

_logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


class OpenVinoCrisperWhisperAdapter(TranscriptionEngine):
    """Batch CrisperWhisper on the NPU. One call at a time, like every adapter here."""

    def __init__(self, model_dir: Path, language: str, mode: CrisperWhisperMode = "intended") -> None:
        self._model_dir = model_dir
        self._language = language
        self._mode = mode
        self._whisper: OpenVinoWhisper | None = None
        self._tokens: WhisperTokens | None = None

    @property
    def encoder_device(self) -> str | None:
        return self._whisper.encoder_device if self._whisper else None

    @property
    def device_note(self) -> str | None:
        if self._whisper and self._whisper.encoder_device == "CPU":
            return "CrisperWhisper's encoder runs on the CPU: the NPU refused it."
        return None

    def load_model(self) -> None:
        started_at = time.perf_counter()
        whisper = OpenVinoWhisper(self._model_dir)
        whisper.load()
        self._tokens = WhisperTokens(self._model_dir)
        self._whisper = whisper
        _logger.info("[load_model openvino crisperwhisper] complete in %.2fs", time.perf_counter() - started_at)

    def is_model_loaded(self) -> bool:
        return self._whisper is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if self._whisper is None:
            raise TranscriptionError("Model not loaded")
        if audio.sample_rate != _SAMPLE_RATE:
            raise TranscriptionError(f"CrisperWhisper needs {_SAMPLE_RATE} Hz audio, got {audio.sample_rate}")
        started_at = time.perf_counter()
        text = transcribe_in_windows(audio, self._generate_text)
        return TranscriptionResult(
            text=text, language=self._language, language_probability=0.0,
            transcription_seconds=time.perf_counter() - started_at,
        )

    def _generate_text(self, window: AudioSegment, context: str) -> str:
        if self._whisper is None or self._tokens is None:
            raise TranscriptionError("Model not loaded")
        prompt = self._tokens.decoder_prompt(self._language, self._mode, context)
        ids = self._whisper.generate(np.ascontiguousarray(window.data, dtype=np.float32), prompt)
        return self._tokens.decode(ids)

    def dispose(self) -> None:
        if self._whisper is not None:
            self._whisper.dispose()
        self._whisper = None
        self._tokens = None
