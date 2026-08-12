"""GPU-only streaming engine for the cache-aware nemotron Parakeet model.

Mirrors :class:`StreamingNanoParakeetEngine`'s shape (GPU-only, warmup, a
``model_factory`` seam for tests) but loads the clean-room cache-aware model
and hands out :class:`CacheAwareStreamingSession` sessions. Runs in fp32 - bf16
hangs MIOpen kernel-tuning on the current RDNA4 + Windows ROCm stack.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.transcription.streaming.streaming_engine import (
    StreamingTranscriptionEngine, StreamingTranscriptionSession,
)
from voicetotext.transcription.streaming.cache_aware.streaming_session import (
    CacheAwareStreamingSession,
)

_SAMPLE_RATE = 16000
_WARMUP_SECONDS = 1.2


class CacheAwareStreamingEngine(StreamingTranscriptionEngine):
    """Loads nemotron-speech-streaming and produces cache-aware sessions."""

    def __init__(
        self,
        repo_id: str,
        model_dir: Path,
        device: str = "cuda",
        model_factory: Callable[[str], object] | None = None,
    ) -> None:
        self._repo_id = repo_id
        self._model_dir = Path(model_dir)
        self._device = device
        self._model_factory = model_factory
        self._model: object | None = None

    def load_model(self) -> None:
        self._require_gpu()
        self._model = self._build_model()

    def _require_gpu(self) -> None:
        if self._device != "cuda":
            raise ValueError(f"Streaming is GPU-only; got device={self._device!r}")

    def _build_model(self) -> object:
        if self._model_factory is not None:
            return self._model_factory(self._device)
        import huggingface_hub as hf

        from voicetotext.transcription.streaming.cache_aware.loader import (
            load_streaming_model,
        )

        filename = f"{self._repo_id.split('/')[-1]}.nemo"
        nemo_path = hf.hf_hub_download(
            repo_id=self._repo_id, filename=filename,
            cache_dir=str(self._model_dir), local_files_only=True,
        )
        return load_streaming_model(nemo_path, device=self._device)

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def warmup(self) -> None:
        """Run one silent session so the first real stream is warm."""
        if not self.is_model_loaded():
            return
        session = self.start_session()
        silence = np.zeros(int(_WARMUP_SECONDS * _SAMPLE_RATE), dtype=np.float32)
        session.feed(silence)
        session.finalize()

    def start_session(self) -> StreamingTranscriptionSession:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")
        return CacheAwareStreamingSession(self._model)

    def dispose(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._release_cache()

    def _release_cache(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
