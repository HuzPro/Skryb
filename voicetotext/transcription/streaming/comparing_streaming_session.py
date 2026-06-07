"""Decorator: live streaming plus an end-of-utterance batch A/B comparison.

TEMPORARY testing aid. Wraps a real streaming session so live dictation is
byte-for-byte unchanged (pure delegation), captures every fed frame, and at
finalize runs a batch reference engine over the FULL captured audio — logging
the streaming transcript next to the batch one so divergence is visible in the
terminal. The batch pass is best-effort: any failure is logged and the real
streaming result is returned untouched.
"""
from __future__ import annotations

import logging

import numpy as np

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionSession, StreamingUpdate,
)


_logger = logging.getLogger(__name__)
_SAMPLE_RATE = 16000  # streaming feeds float32 16 kHz mono frames


class ComparingStreamingSession(StreamingTranscriptionSession):
    """Delegates live streaming, then logs a batch transcript for comparison."""

    def __init__(
        self,
        inner: StreamingTranscriptionSession,
        batch: TranscriptionEngine | None,
    ) -> None:
        self._inner = inner
        self._batch = batch
        self._captured: list[np.ndarray] = []

    def feed(self, chunk: np.ndarray) -> StreamingUpdate:
        self._captured.append(chunk.astype(np.float32))  # astype copies
        return self._inner.feed(chunk)

    def finalize(self) -> TranscriptionResult:
        streaming_result = self._inner.finalize()
        self._log_comparison_safely(streaming_result)
        return streaming_result

    def _log_comparison_safely(self, streaming: TranscriptionResult) -> None:
        if self._batch is None:
            return
        try:
            self._log_comparison(streaming)
        except Exception as exc:  # noqa: BLE001 — comparison is best-effort
            _logger.warning("Batch comparison failed (non-fatal): %s", exc)

    def _log_comparison(self, streaming: TranscriptionResult) -> None:
        segment = self._captured_segment()
        if segment is None:
            return
        batch = self._batch.transcribe(segment)
        _log_side_by_side(
            streaming.text, batch.text, streaming.transcription_seconds,
        )

    def _captured_segment(self) -> AudioSegment | None:
        if not self._captured:
            return None
        data = np.concatenate(self._captured)
        if data.size == 0:
            return None
        return AudioSegment(data=data, sample_rate=_SAMPLE_RATE)


def _log_side_by_side(
    streaming_text: str, batch_text: str, seconds: float,
) -> None:
    verdict = "identical" if _same(streaming_text, batch_text) else "differs"
    _logger.info(
        "=== streaming vs batch (%s) in %.2fs ===\n  streaming: %s\n  batch    : %s",
        verdict, seconds, streaming_text, batch_text,
    )


def _same(left: str, right: str) -> bool:
    return left.strip() == right.strip()
