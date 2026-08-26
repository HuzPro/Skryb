import logging
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.backend_imports import (
    make_engine_dependency_error,
)
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_diagnostics import vram_snapshot
from voicetotext.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)


_CPU_DEVICE = "cpu"
# The model card recommends 256 new tokens per transcription request;
# push-to-talk utterances (2-30 s) fit comfortably under that bound and
# a smaller cap keeps a runaway generation from stalling the hotkey.
_MAX_NEW_TOKENS = 256


class _Qwen3AsrProcessor(Protocol):
    def apply_transcription_request(
        self, audio: object, language: str,
    ) -> object: ...

    def decode(self, token_ids: object, return_format: str) -> str: ...


class _Qwen3AsrModel(Protocol):
    def generate(self, **inputs: object) -> object: ...


class Qwen3AsrAdapter(TranscriptionEngine):
    """Qwen3-ASR engine backed by native transformers (PyTorch).

    Loads the ``Qwen/Qwen3-ASR-*-hf`` checkpoints (plain transformers
    weights, no remote code) from the app's models dir. Works on any
    GPU that PyTorch supports, including AMD via ROCm.
    """

    def __init__(
        self,
        repo_id: str,
        device: str,
        model_dir: Path,
        language: str,
    ) -> None:
        self._repo_id = repo_id
        self._device = device
        self._model_dir = model_dir
        self._language = language
        self._runtime: _Qwen3AsrRuntime | None = None

    def load_model(self) -> None:
        self._log_phase("load_model", f"start device={self._device}")
        started_at = time.monotonic()
        self._runtime = _load_runtime(
            self._repo_id, self._model_dir, self._device,
        )
        self._log_timed_phase("load_model", "complete", started_at)

    def is_model_loaded(self) -> bool:
        return self._runtime is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        runtime = self._require_runtime()
        start = time.monotonic()
        text = runtime.transcribe(audio.data, self._language)
        return TranscriptionResult(
            text=text.strip(),
            language=self._language,
            language_probability=0.0,
            transcription_seconds=time.monotonic() - start,
        )

    def _require_runtime(self) -> "_Qwen3AsrRuntime":
        if self._runtime is None:
            raise TranscriptionError("Model not loaded")
        return self._runtime

    def dispose(self) -> None:
        if self._runtime is None:
            return
        self._log_phase("dispose", f"start device={self._device}")
        self._runtime = None
        self._log_phase("dispose", "refs nulled")
        if self._device != _CPU_DEVICE:
            import torch
            torch.cuda.empty_cache()
        self._log_phase("dispose", "cache released")

    def _log_phase(self, operation: str, message: str) -> None:
        _logger.info(
            "[%s %s] %s | %s",
            operation, self._repo_id, message, vram_snapshot(),
        )

    def _log_timed_phase(
        self, operation: str, message: str, started_at: float,
    ) -> None:
        elapsed = time.monotonic() - started_at
        _logger.info(
            "[%s %s] %s in %.2fs | %s",
            operation, self._repo_id, message, elapsed, vram_snapshot(),
        )


class _Qwen3AsrRuntime:
    """A loaded processor + model pair that turns waveforms into text.

    Decoding is greedy (the checkpoint's generation config sets
    ``do_sample=False``) and the configured language is forced, so the
    assistant turn is prefilled with ``language <NAME><asr_text>`` and
    the model generates only the transcription text - no language tag
    left to parse out of the output.
    """

    def __init__(
        self,
        processor: _Qwen3AsrProcessor,
        model: _Qwen3AsrModel,
        device: str,
        dtype: object,
    ) -> None:
        self._processor = processor
        self._model = model
        self._device = device
        self._dtype = dtype

    def transcribe(self, waveform: np.ndarray, language: str) -> str:
        inputs = self._processor.apply_transcription_request(
            audio=waveform, language=language,
        ).to(self._device, self._dtype)
        prompt_length = inputs["input_ids"].shape[1]
        output_ids = self._model.generate(
            **inputs, max_new_tokens=_MAX_NEW_TOKENS,
        )
        generated = output_ids[0, prompt_length:]
        return self._processor.decode(
            generated, return_format="transcription_only",
        )


def _load_runtime(
    repo_id: str, model_dir: Path, device: str,
) -> _Qwen3AsrRuntime:
    transformers_module, torch_module = _import_backend()
    dtype = _dtype_for(torch_module, device)
    cache_dir = str(model_dir)
    processor = transformers_module.AutoProcessor.from_pretrained(
        repo_id, cache_dir=cache_dir, local_files_only=True,
    )
    model = transformers_module.Qwen3ASRForConditionalGeneration.from_pretrained(
        repo_id,
        cache_dir=cache_dir,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return _Qwen3AsrRuntime(
        processor=processor, model=model, device=device, dtype=dtype,
    )


def _dtype_for(torch_module: object, device: str) -> object:
    """bf16 on GPU (the checkpoint's native dtype), fp32 on CPU.

    The safetensors weights are stored in bfloat16, so bf16 on GPU is a
    straight load with no precision loss; CPU has no fast bf16 matmul,
    so fp32 is both correct and faster there.
    """
    if device == _CPU_DEVICE:
        return torch_module.float32
    return torch_module.bfloat16


def _import_backend() -> tuple[object, object]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="transformers",
            install_hint=(
                'transformers not installed: pip install "transformers>=5.13"'
            ),
            underlying=exc,
        ) from exc
    return transformers, torch
