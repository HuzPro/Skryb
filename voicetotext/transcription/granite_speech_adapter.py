import logging
import math
import time
from pathlib import Path

import numpy as np

from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    TranscriptionError,
)
from voicetotext.transcription.backend_imports import (
    make_engine_dependency_error,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_diagnostics import vram_snapshot
from voicetotext.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)

_REQUIRED_SAMPLE_RATE = 16000

# Output-token budget scales with audio length. 12 tok/s is ~2x the
# observed Granite token rate for English speech (~5–6 tok/s), so the
# generator never runs out of budget before a chunk is fully
# transcribed. The floor protects sub-second utterances; the ceiling
# bounds pathological inputs.
_TOKENS_PER_SECOND = 12
_TOKEN_BUDGET_FLOOR = 128
_TOKEN_BUDGET_CEILING = 8192

# Chunking. IBM's Granite 3.3 announcement gives ~1 minute per segment
# as the supported audio length for the base 4.1-2b model; the 4.1-2b-
# plus variant is the one trained on long-form. 55 s leaves headroom
# under the 60 s guideline and keeps each chunk well under the 4096-
# token LLM context (each audio second expands to ~10 LLM tokens after
# the q-former).
_MAX_CHUNK_SECONDS = 55.0
_SILENCE_SEARCH_RADIUS_SECONDS = 5.0
_SILENCE_WINDOW_SECONDS = 0.1

# Inference recipe from the Granite Speech tech report (arXiv
# 2505.08699). The IBM team documented that the model card's quickstart
# (greedy, no penalty) is not the production recipe - without these
# guards the model collapses into a repetition loop on audio outside
# its training distribution. ``num_beams=4`` is IBM's recommendation;
# users can override via ``transcription.beam_size`` in config to
# trade accuracy for ~Nx decoder speedup.
_DEFAULT_NUM_BEAMS = 4
_REPETITION_PENALTY = 3.0

# Repetition-warning heuristic. If the same N-word window appears
# ``_REPETITION_THRESHOLD`` or more times in a chunk's output, log a
# warning so silent regressions are debuggable.
_REPETITION_WINDOW_WORDS = 6
_REPETITION_THRESHOLD = 3

_PROMPT = (
    "<|audio|>transcribe the speech with proper punctuation "
    "and capitalization."
)


def _token_budget_for_duration(duration_seconds: float) -> int:
    scaled = math.ceil(duration_seconds * _TOKENS_PER_SECOND)
    return max(_TOKEN_BUDGET_FLOOR, min(_TOKEN_BUDGET_CEILING, scaled))


def _find_quietest_offset(
    data: np.ndarray, target_sample: int, sample_rate: int,
) -> int:
    """Search ±_SILENCE_SEARCH_RADIUS_SECONDS around ``target_sample``
    for the lowest-RMS 100 ms window and return the midpoint of that
    window. Snapping chunk boundaries into quiet regions avoids cutting
    mid-word, which Granite cannot recover from across chunks.
    """
    radius = int(_SILENCE_SEARCH_RADIUS_SECONDS * sample_rate)
    window = int(_SILENCE_WINDOW_SECONDS * sample_rate)
    start = max(0, target_sample - radius)
    end = min(len(data) - window, target_sample + radius)
    if end <= start:
        return target_sample
    step = max(1, window // 4)
    quietest_offset = target_sample
    quietest_energy = float("inf")
    for offset in range(start, end, step):
        chunk = data[offset:offset + window]
        energy = float(np.sqrt(np.mean(chunk * chunk)))
        if energy < quietest_energy:
            quietest_energy = energy
            quietest_offset = offset + window // 2
    return quietest_offset


def _split_into_chunks(audio: AudioSegment) -> list[AudioSegment]:
    if audio.duration_seconds <= _MAX_CHUNK_SECONDS:
        return [audio]
    sample_rate = audio.sample_rate
    chunk_samples = int(_MAX_CHUNK_SECONDS * sample_rate)
    chunks: list[AudioSegment] = []
    cursor = 0
    total = len(audio.data)
    while cursor < total:
        target_end = min(total, cursor + chunk_samples)
        if target_end < total:
            target_end = _find_quietest_offset(
                audio.data, target_end, sample_rate,
            )
        chunk_data = np.ascontiguousarray(audio.data[cursor:target_end])
        chunks.append(AudioSegment(data=chunk_data, sample_rate=sample_rate))
        cursor = target_end
    return chunks


def _looks_repetitive(text: str) -> bool:
    words = text.split()
    if len(words) < _REPETITION_WINDOW_WORDS * _REPETITION_THRESHOLD:
        return False
    counts: dict[str, int] = {}
    for i in range(len(words) - _REPETITION_WINDOW_WORDS + 1):
        window = " ".join(words[i:i + _REPETITION_WINDOW_WORDS])
        counts[window] = counts.get(window, 0) + 1
        if counts[window] >= _REPETITION_THRESHOLD:
            return True
    return False


class _GeneratedOnlyRepetitionPenalty:
    """Repetition penalty scoped to tokens emitted after the prompt.

    Transformers' built-in ``RepetitionPenaltyLogitsProcessor`` also
    penalises prompt tokens. For Granite Speech, the prompt contains
    the expanded audio placeholder tokens - penalising those treats
    legitimate audio context as "already used" and degrades quality.
    The IBM team confirmed this in the public granite-speech-3.3-8b
    discussion; we apply the penalty only to the generated slice.
    """

    def __init__(self, penalty: float, prompt_length: int) -> None:
        if penalty <= 0:
            raise ValueError("penalty must be positive")
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        self._penalty = penalty
        self._prompt_length = prompt_length

    def __call__(self, input_ids: object, scores: object) -> object:
        import torch

        if input_ids.shape[-1] <= self._prompt_length:
            return scores
        generated_ids = input_ids[:, self._prompt_length:]
        gathered = torch.gather(scores, 1, generated_ids)
        adjusted = torch.where(
            gathered < 0,
            gathered * self._penalty,
            gathered / self._penalty,
        )
        return scores.scatter(1, generated_ids, adjusted)


class GraniteSpeechAdapter(TranscriptionEngine):
    """Granite Speech 4.1 engine backed by transformers' Auto-classes.

    Granite is an autoregressive multimodal LLM - it does ASR via
    ``model.generate()`` over a chat-template prompt that contains the
    ``<|audio|>`` token. The base 4.1-2b variant is only trained on
    ≤60 s utterances, so long audio is split into ≤55 s chunks at
    silence-aware boundaries and transcribed sequentially.
    """

    def __init__(
        self,
        repo_id: str,
        device: str,
        model_dir: Path,
        dtype: str = "bfloat16",
        num_beams: int = _DEFAULT_NUM_BEAMS,
    ) -> None:
        if num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {num_beams}")
        self._repo_id = repo_id
        self._device = device
        self._model_dir = model_dir
        self._dtype_name = dtype
        self._num_beams = num_beams
        self._processor: object | None = None
        self._model: object | None = None

    def load_model(self) -> None:
        torch_module, auto_model_class, auto_processor_class = self._import_backend()

        cache_dir = str(self._model_dir)
        torch_dtype = getattr(torch_module, self._dtype_name)
        self._log_phase(
            "load_model",
            f"start device={self._device} dtype={self._dtype_name}",
        )
        started_at = time.monotonic()
        self._processor = auto_processor_class.from_pretrained(
            self._repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        self._model = auto_model_class.from_pretrained(
            self._repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
            device_map=self._device,
            dtype=torch_dtype,
        )
        self._log_timed_phase("load_model", "complete", started_at)

    def _import_backend(self) -> tuple[object, object, object]:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise make_engine_dependency_error(
                package_name="transformers",
                install_hint="transformers not installed: pip install transformers",
                underlying=exc,
            ) from exc
        return torch, AutoModelForSpeechSeq2Seq, AutoProcessor

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")
        if audio.sample_rate != _REQUIRED_SAMPLE_RATE:
            raise TranscriptionError(
                f"Granite Speech requires 16000 Hz audio, "
                f"got {audio.sample_rate}"
            )
        chunks = _split_into_chunks(audio)
        return self._transcribe_chunks(chunks)

    def _transcribe_chunks(
        self, chunks: list[AudioSegment],
    ) -> TranscriptionResult:
        started_at = time.monotonic()
        parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            text = self._transcribe_chunk(chunk, index, len(chunks))
            if text:
                parts.append(text)
        return TranscriptionResult(
            text=" ".join(parts),
            language="en",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - started_at,
        )

    def _transcribe_chunk(
        self, audio: AudioSegment, position: int, total: int,
    ) -> str:
        import torch

        inputs = self._prepare_inputs(audio)
        n_input_tokens = inputs["input_ids"].shape[-1]
        token_budget = _token_budget_for_duration(audio.duration_seconds)
        generate_kwargs = self._build_generate_kwargs(
            token_budget, n_input_tokens,
        )
        with torch.no_grad():
            out = self._model.generate(**inputs, **generate_kwargs)
        generated_ids = out[:, n_input_tokens:]
        text = self._processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()
        self._log_chunk_usage(
            position, total, audio.duration_seconds, token_budget, generated_ids,
        )
        self._warn_if_repetitive(text, position, total)
        return text

    def _prepare_inputs(self, audio: AudioSegment) -> object:
        import torch

        wav = torch.from_numpy(audio.data).unsqueeze(0)
        chat = [{"role": "user", "content": _PROMPT}]
        prompt = self._processor.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self._processor(
            prompt,
            wav,
            device=self._device,
            return_tensors="pt",
        ).to(self._device)

    def _build_generate_kwargs(
        self, token_budget: int, n_input_tokens: int,
    ) -> dict[str, object]:
        penalty = _GeneratedOnlyRepetitionPenalty(
            penalty=_REPETITION_PENALTY, prompt_length=n_input_tokens,
        )
        return {
            "max_new_tokens": token_budget,
            "do_sample": False,
            "num_beams": self._num_beams,
            "logits_processor": [penalty],
        }

    def _log_chunk_usage(
        self,
        position: int,
        total: int,
        audio_seconds: float,
        budget: int,
        generated_ids: object,
    ) -> None:
        _logger.info(
            "[transcribe %s] chunk %d/%d audio=%.1fs "
            "tokens_budget=%d tokens_used=%s",
            self._repo_id, position, total, audio_seconds, budget,
            getattr(generated_ids, "shape", ["?"])[-1],
        )

    def _warn_if_repetitive(
        self, text: str, position: int, total: int,
    ) -> None:
        if not _looks_repetitive(text):
            return
        _logger.warning(
            "[transcribe %s] chunk %d/%d output is repetitive "
            "(possible decoder collapse): %s",
            self._repo_id, position, total, text[:160],
        )

    def dispose(self) -> None:
        if self._model is None:
            return
        self._log_phase("dispose", f"start device={self._device}")
        self._processor = None
        self._model = None
        self._log_phase("dispose", "refs nulled")
        if self._device != "cpu":
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
