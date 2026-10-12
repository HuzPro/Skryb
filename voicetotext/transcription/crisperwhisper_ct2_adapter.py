"""CrisperWhisper 2.0 inference through the CTranslate2 runtime."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

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
    transcribe_in_windows,
)
from voicetotext.transcription.whisper_tokens import WhisperTokens


if TYPE_CHECKING:
    from ctranslate2.models import Whisper


_logger = logging.getLogger(__name__)

_CPU_DEVICE = "cpu"
_DECODER_POSITIONS = 448
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


_T = TypeVar("_T")


class _ModelThread:
    """The thread a CTranslate2 model is created on, kept alive until it is freed.

    CTranslate2 4.8.2 (ROCm build, Windows) frees a model's memory through
    state that belongs to the thread that created the model. Once that
    thread has exited, freeing the model corrupts the heap and the process
    dies (0xC0000374 or 0xC0000005, inside ctranslate2.dll). Freeing it from
    any thread is safe while the creating thread is still alive. Engines
    used to be created on loader threads that exit right after loading, so
    every model swap, quit cleanup and interpreter exit could crash.

    Daemon, so a model still loaded at interpreter exit is freed while this
    thread still exists rather than after it has been joined.
    """

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue[
            tuple[Callable[[], object], Future[object]] | None
        ] = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._serve, daemon=True, name="ct2-model",
        )
        self._thread.start()

    def call(self, job: Callable[[], _T]) -> _T:
        """Run *job* on this thread and return its result or raise its error."""
        done: Future[object] = Future()
        self._jobs.put((job, done))
        return done.result()  # type: ignore[return-value]

    def close(self) -> None:
        """Let the thread end; only once the model it created is freed."""
        self._jobs.put(None)

    def _serve(self) -> None:
        while (item := self._jobs.get()) is not None:
            job, done = item
            try:
                done.set_result(job())
            except BaseException as exc:  # noqa: BLE001 - re-raised by call()
                done.set_exception(exc)


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
        self._model_thread: _ModelThread | None = None

    def load_model(self) -> None:
        started_at = time.monotonic()
        self.dispose()
        whisper_type = _import_backend()
        tokens = WhisperTokens(self._model_dir)
        model_thread = _ModelThread()
        try:
            model = model_thread.call(
                lambda: whisper_type(
                    str(self._model_dir),
                    device=self._device,
                    compute_type=self._compute_type(),
                ),
            )
        except BaseException:
            model_thread.close()
            raise
        self._model_thread = model_thread
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
        text = transcribe_in_windows(audio, self._generate_text)
        return TranscriptionResult(
            text=text,
            language=self._language,
            language_probability=0.0,
            transcription_seconds=time.monotonic() - started_at,
        )

    def _generate_text(self, audio: AudioSegment, context: str) -> str:
        model, tokens, _ = self._loaded()
        prompt = tokens.decoder_prompt(self._language, self._mode, context)
        features = np.ascontiguousarray(
            log_mel_spectrogram(audio)[np.newaxis, :, :],
        )
        results = model.generate(
            _storage_view(features),
            [prompt],
            beam_size=1,
            sampling_topk=1,
            suppress_blank=False,
            max_length=_max_length(prompt),
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
        model_thread, self._model_thread = self._model_thread, None
        model, self._model = [self._model], None
        self._tokens = None
        self._prompt = None
        if model_thread is None:
            return
        # The last reference goes on the thread that created the model,
        # and only then may that thread end (see ``_ModelThread``).
        model_thread.call(model.clear)
        model_thread.close()


def _max_length(prompt: list[int]) -> int:
    """Budget that keeps the prompt plus generation inside 448 positions.

    CTranslate2 halves whatever budget it is given
    (min(max_length / 2, max_length - start_step)), so this is double the
    new tokens the prompt leaves room for. The prompt is longer on every
    window after the first, by the continuation context it carries.
    """
    return 2 * (_DECODER_POSITIONS - len(prompt) - 1)


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
