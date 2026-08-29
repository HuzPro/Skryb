import logging
import time
from pathlib import Path

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


_SAMPLE_RATE = 16000
# The model card's reference inference caps audio at 30 s
# (``audio_max_length=30 * 16000``) - the Whisper-style encoder's
# receptive field. Longer recordings are chunked so nothing is
# silently truncated by the feature extractor.
_CHUNK_SECONDS = 30.0
_MAX_NEW_TOKENS = 256
_TRANSCRIBE_PROMPT = "Please transcribe this audio."
_CPU_DEVICE = "cpu"


def _build_bad_words_ids(tokenizer: object) -> list[list[int]]:
    """Single-token sequences generation must never emit.

    From the model card's reference inference: ban every special or
    ``<...>`` control token except EOS, so greedy decoding cannot
    wander into the model's non-ASR control vocabulary.
    """
    eos_ids = tokenizer.eos_token_id
    keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    bad_ids = set(tokenizer.all_special_ids) - keep_ids
    bad_ids.update(
        token_id
        for token, token_id in tokenizer.get_added_vocab().items()
        if token.startswith("<")
        and token.endswith(">")
        and token_id not in keep_ids
    )
    return [[token_id] for token_id in sorted(bad_ids)]


class _ArkAsrBackend:
    """One loaded ARK-ASR stack: model, processor, and tokenizer.

    Owns the per-utterance prompt-build -> generate -> decode sequence
    so the adapter above it only deals in waveforms and text.
    """

    def __init__(
        self,
        model: object,
        processor: object,
        tokenizer: object,
        torch_module: object,
        device: str,
        dtype: object,
    ) -> None:
        self._model = model
        self._processor = processor
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._device = device
        self._dtype = dtype
        self._bad_words_ids = _build_bad_words_ids(tokenizer)

    def transcribe(self, waveform: np.ndarray) -> str:
        inputs = self._prepare_inputs(waveform)
        outputs = self._generate(inputs)
        return self._decode(inputs, outputs)

    def _prepare_inputs(self, waveform: np.ndarray) -> object:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "array": waveform},
                    {"type": "text", "text": _TRANSCRIBE_PROMPT},
                ],
            },
        ]
        inputs = self._processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            sampling_rate=_SAMPLE_RATE,
            audio_padding="longest",
            text_kwargs={"padding": "longest"},
            audio_max_length=int(_CHUNK_SECONDS * _SAMPLE_RATE),
        ).to(self._device)
        inputs["audios"] = inputs["audios"].to(dtype=self._dtype)
        return inputs

    def _generate(self, inputs: object) -> object:
        with self._torch.inference_mode():
            return self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=_MAX_NEW_TOKENS,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                bad_words_ids=self._bad_words_ids,
            )

    def _decode(self, inputs: object, outputs: object) -> str:
        prompt_length = inputs["input_ids"].shape[1]
        decoded = self._tokenizer.batch_decode(
            outputs[:, prompt_length:],
            skip_special_tokens=True,
        )
        return str(decoded[0])


class ArkAsrAdapter(TranscriptionEngine):
    """ARK-ASR-3B engine backed by HuggingFace Transformers remote code.

    Whisper-style RoPE audio encoder -> MLP adapter -> Qwen decoder
    (``Audio8/ARK-ASR-3B``). Runs BF16 with SDPA attention on GPU
    (CUDA/ROCm), FP32 on CPU.

    *revision* is the audited commit from ``ModelInfo.revision`` - the
    same one the downloader fetched, so load and download can never
    disagree about which code ``trust_remote_code`` executes. Note what
    the pin does and does not buy: it selects *which* commit is
    requested. It does not verify the bytes that arrive, which rest on
    TLS alone.
    """

    def __init__(
        self,
        repo_id: str,
        revision: str,
        device: str,
        model_dir: Path,
    ) -> None:
        self._repo_id = repo_id
        self._revision = revision
        self._device = device
        self._model_dir = model_dir
        self._backend: _ArkAsrBackend | None = None

    def load_model(self) -> None:
        self._log_phase("load_model", f"start device={self._device}")
        started_at = time.monotonic()
        transformers_module, torch_module = self._import_backend()
        snapshot_dir = self._resolve_pinned_snapshot()
        self._backend = self._load_backend(
            transformers_module, torch_module, snapshot_dir,
        )
        self._log_timed_phase("load_model", "complete", started_at)

    def _import_backend(self) -> tuple[object, object]:
        try:
            import librosa
            import soundfile
            import torch
            import transformers
        except ImportError as exc:
            raise make_engine_dependency_error(
                package_name=exc.name or "transformers",
                install_hint=(
                    "ARK-ASR backend not installed: "
                    "pip install transformers librosa soundfile"
                ),
                underlying=exc,
            ) from exc
        del librosa, soundfile
        return transformers, torch

    def _resolve_pinned_snapshot(self) -> str:
        from huggingface_hub import snapshot_download

        return str(
            snapshot_download(
                repo_id=self._repo_id,
                revision=self._revision,
                cache_dir=str(self._model_dir),
                local_files_only=True,
            )
        )

    def _load_backend(
        self,
        transformers_module: object,
        torch_module: object,
        snapshot_dir: str,
    ) -> _ArkAsrBackend:
        dtype = self._dtype_for(torch_module)
        model = transformers_module.AutoModelForCausalLM.from_pretrained(
            snapshot_dir,
            trust_remote_code=True,
            dtype=dtype,
            attn_implementation="sdpa",
        ).to(self._device)
        model.eval()
        processor = transformers_module.AutoProcessor.from_pretrained(
            snapshot_dir, trust_remote_code=True,
        )
        tokenizer = transformers_module.AutoTokenizer.from_pretrained(
            snapshot_dir, trust_remote_code=True,
        )
        return _ArkAsrBackend(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            torch_module=torch_module,
            device=self._device,
            dtype=dtype,
        )

    def _dtype_for(self, torch_module: object) -> object:
        if self._device == _CPU_DEVICE:
            return torch_module.float32
        return torch_module.bfloat16

    def is_model_loaded(self) -> bool:
        return self._backend is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        self._raise_if_not_loaded()
        start = time.monotonic()
        text = self._transcribe_chunked(audio)
        return TranscriptionResult(
            text=text,
            language="en",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - start,
        )

    def _raise_if_not_loaded(self) -> None:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")

    def _transcribe_chunked(self, audio: AudioSegment) -> str:
        chunks = self._split_into_chunks(audio)
        parts = [
            self._backend.transcribe(chunk.data).strip()
            for chunk in chunks
        ]
        return " ".join(part for part in parts if part)

    @staticmethod
    def _split_into_chunks(audio: AudioSegment) -> list[AudioSegment]:
        samples_per_chunk = int(_CHUNK_SECONDS * audio.sample_rate)
        total_samples = audio.sample_count
        chunks: list[AudioSegment] = []
        for start in range(0, total_samples, samples_per_chunk):
            end = min(start + samples_per_chunk, total_samples)
            chunk_data = audio.data[start:end]
            if chunk_data.size == 0:
                continue
            chunks.append(
                AudioSegment(data=chunk_data, sample_rate=audio.sample_rate),
            )
        return chunks

    def dispose(self) -> None:
        if self._backend is None:
            return
        self._log_phase("dispose", f"start device={self._device}")
        self._backend = None
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
