"""Decorator: a streaming engine that also loads a batch A/B reference.

TEMPORARY testing aid. Wraps a real streaming engine and owns a batch
:class:`TranscriptionEngine`; ``load_model`` loads both and ``start_session``
hands out a :class:`ComparingStreamingSession`. The batch reference is
optional — if it fails to load (e.g. its weights are not downloaded), the live
streaming engine keeps working and the comparison is silently disabled.
"""
from __future__ import annotations

import logging

from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionEngine, StreamingTranscriptionSession,
)
from voicetotext.transcription.streaming.comparing_streaming_session import (
    ComparingStreamingSession,
)


_logger = logging.getLogger(__name__)


class ComparingStreamingEngine(StreamingTranscriptionEngine):
    """Live streaming engine wrapped to also run a batch reference per dictation."""

    def __init__(
        self,
        streaming: StreamingTranscriptionEngine,
        batch: TranscriptionEngine,
    ) -> None:
        self._streaming = streaming
        self._batch: TranscriptionEngine | None = batch

    def load_model(self) -> None:
        self._streaming.load_model()
        self._load_batch_reference()

    def _load_batch_reference(self) -> None:
        try:
            self._batch.load_model()
        except Exception as exc:  # noqa: BLE001 — comparison is optional
            _logger.warning(
                "Batch comparison disabled (reference model unavailable): %s", exc,
            )
            self._batch = None

    def is_model_loaded(self) -> bool:
        return self._streaming.is_model_loaded()

    def reports_own_transcript(self) -> bool:
        # The comparison session logs the streaming transcript next to the
        # batch one, so the use case must not also log the generic line.
        return True

    def warmup(self) -> None:
        # Only the live engine is warmed; the batch reference pays its
        # cold-start on the first finalize (off the typing path).
        self._streaming.warmup()

    def start_session(self) -> StreamingTranscriptionSession:
        return ComparingStreamingSession(self._streaming.start_session(), self._batch)

    def dispose(self) -> None:
        self._streaming.dispose()
        if self._batch is not None:
            self._batch.dispose()
