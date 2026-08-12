import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.transcription.parakeet_loader import ParakeetModelLoader
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.local_agreement import (
    LocalAgreementPolicy,
)
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionEngine,
    StreamingTranscriptionSession,
    StreamingUpdate,
)


_logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
# Re-decode the whole window every ~0.6 s of new audio. Parakeet is
# hundreds x realtime, so this is cheap; smaller values feel snappier but
# spend more GPU on redundant re-encodes.
_CADENCE_SECONDS = 0.6
# Once the live window passes this, trim everything up to the last
# committed word so the full-context encoder never re-runs on an
# unbounded buffer (the O(n^2) guard for long dictation).
_MAX_WINDOW_SECONDS = 30.0
# Silent audio fed at load time to force one decode and compile kernels.
_WARMUP_SECONDS = 0.6
_WARMUP_CHUNKS = 3


def _join(prefix: str, suffix: str) -> str:
    return " ".join(part for part in (prefix, suffix) if part)


class _ParakeetStreamingSession(StreamingTranscriptionSession):
    """Buffered streaming over an offline Parakeet model.

    The full-context encoder cannot carry state, so each tick re-decodes
    the whole live window. :class:`LocalAgreementPolicy` commits the
    stable prefix (never retracted); committed audio is trimmed away once
    the window grows too long, keeping per-tick cost bounded.
    """

    def __init__(
        self,
        model: object,
        sample_rate: int = _SAMPLE_RATE,
        cadence_seconds: float = _CADENCE_SECONDS,
        max_window_seconds: float = _MAX_WINDOW_SECONDS,
    ) -> None:
        self._model = model
        self._sample_rate = sample_rate
        self._cadence_samples = int(cadence_seconds * sample_rate)
        self._max_window_samples = int(max_window_seconds * sample_rate)
        self._window_chunks: list[np.ndarray] = []
        self._window_samples = 0
        self._samples_since_decode = 0
        self._policy = LocalAgreementPolicy()
        self._committed_prefix = ""
        self._tentative = ""
        self._started_at = time.monotonic()

    def feed(self, chunk: np.ndarray) -> StreamingUpdate:
        self._append(chunk)
        if self._samples_since_decode >= self._cadence_samples:
            self._samples_since_decode = 0
            self._decode_and_commit()
        return self._snapshot(is_final=False)

    def finalize(self) -> TranscriptionResult:
        if self._window_chunks:
            words, _ = self._decode_window()
            self._policy.flush(words)
        return TranscriptionResult(
            text=self._full_committed(),
            language="en",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - self._started_at,
        )

    def _append(self, chunk: np.ndarray) -> None:
        self._window_chunks.append(chunk)
        self._window_samples += len(chunk)
        self._samples_since_decode += len(chunk)

    def _decode_and_commit(self) -> None:
        words, ends = self._decode_window()
        decision = self._policy.observe(words)
        self._tentative = " ".join(decision.tentative)
        self._trim_if_window_too_long(ends)

    def _decode_window(self) -> tuple[list[str], list[float]]:
        window = np.concatenate(self._window_chunks)
        result = self._model.transcribe(window, timestamps=True)
        word_infos = result.timestamp.get("word", [])
        words = [str(info["word"]) for info in word_infos]
        ends = [float(info["end"]) for info in word_infos]
        return words, ends

    def _trim_if_window_too_long(self, ends: list[float]) -> None:
        if self._window_samples <= self._max_window_samples:
            return
        # The policy never retracts, but a later decode can return fewer
        # words than are already committed; only trust end-times for words
        # the current decode actually produced.
        trimmable = min(len(self._policy.committed_words), len(ends))
        if trimmable == 0:
            return
        self._trim_to_committed_word(trimmable, ends)

    def _trim_to_committed_word(
        self, trimmable: int, ends: list[float],
    ) -> None:
        trim_samples = int(ends[trimmable - 1] * self._sample_rate)
        if trim_samples <= 0:
            return
        self._committed_prefix = _join(
            self._committed_prefix, " ".join(self._policy.committed_words),
        )
        remainder = np.concatenate(self._window_chunks)[trim_samples:]
        self._window_chunks = [remainder] if remainder.size else []
        self._window_samples = int(remainder.shape[0])
        self._policy = LocalAgreementPolicy()
        self._tentative = ""

    def _snapshot(self, is_final: bool) -> StreamingUpdate:
        return StreamingUpdate(
            committed_text=self._full_committed(),
            tentative_text=self._tentative,
            is_final=is_final,
        )

    def _full_committed(self) -> str:
        return _join(self._committed_prefix, " ".join(self._policy.committed_words))


class StreamingNanoParakeetEngine(StreamingTranscriptionEngine):
    """GPU-only streaming engine that reuses the downloaded Parakeet model.

    ``model_factory`` exists for tests - production leaves it ``None`` and
    the engine loads via :class:`ParakeetModelLoader`, sharing nano-parakeet's
    in-memory model cache with the batch adapter.
    """

    def __init__(
        self,
        repo_id: str,
        model_dir: Path,
        device: str = "cuda",
        model_factory: Callable[[str], object] | None = None,
        sample_rate: int = _SAMPLE_RATE,
        cadence_seconds: float = _CADENCE_SECONDS,
        max_window_seconds: float = _MAX_WINDOW_SECONDS,
    ) -> None:
        self._repo_id = repo_id
        self._model_dir = model_dir
        self._device = device
        self._model_factory = model_factory
        self._sample_rate = sample_rate
        self._cadence_seconds = cadence_seconds
        self._max_window_seconds = max_window_seconds
        self._model: object | None = None

    def load_model(self) -> None:
        self._require_gpu()
        self._model = self._build_model()

    def _require_gpu(self) -> None:
        if self._device != "cuda":
            raise ValueError(
                f"Streaming is GPU-only; got device={self._device!r}"
            )

    def _build_model(self) -> object:
        if self._model_factory is not None:
            return self._model_factory(self._device)
        return ParakeetModelLoader(self._repo_id, self._model_dir).load(
            self._device,
        )

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        """Run one silent session so the first real stream is warm."""
        if not self.is_model_loaded():
            return
        session = self.start_session()
        silence = np.zeros(
            int(_WARMUP_SECONDS * self._sample_rate), dtype=np.float32,
        )
        for _ in range(_WARMUP_CHUNKS):
            session.feed(silence)
        session.finalize()

    def start_session(self) -> StreamingTranscriptionSession:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")
        return _ParakeetStreamingSession(
            self._model,
            sample_rate=self._sample_rate,
            cadence_seconds=self._cadence_seconds,
            max_window_seconds=self._max_window_seconds,
        )

    def dispose(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._release_cache()

    def _release_cache(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
