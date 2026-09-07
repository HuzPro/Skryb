"""CrisperWhisper 2.0 inference using the installed Transformers backend."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np

from voicetotext.core.config import ConfigValue
from voicetotext.core.exceptions import (
    ConfigurationError,
    TranscriptionError,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.backend_imports import (
    make_engine_dependency_error,
)
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_diagnostics import vram_snapshot
from voicetotext.transcription.engine_types import CrisperWhisperMode
from voicetotext.transcription.result import TranscriptionResult


if TYPE_CHECKING:
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )


_logger = logging.getLogger(__name__)

_CPU_DEVICE = "cpu"
# ``WhisperFeatureExtractor`` truncates to ``chunk_length`` seconds by
# default, so audio past this never reaches the encoder. Longer
# recordings are split rather than silently losing their tail.
_ENCODER_WINDOW_SECONDS = 30.0
# Boundaries are nudged onto the quietest frame in this trailing slice of
# the window so a split lands between words instead of inside one.
_BOUNDARY_SEARCH_SECONDS = 2.0
_BOUNDARY_FRAME_SECONDS = 0.02
_FALLBACK_MAX_NEW_TOKENS = 256
_MODE_TAG_COUNT = 5
_VALID_MODES = frozenset({"intended", "verbatim"})


def split_into_encoder_windows(audio: AudioSegment) -> list[AudioSegment]:
    """Slice *audio* into windows the encoder will accept whole."""
    window_samples = int(_ENCODER_WINDOW_SECONDS * audio.sample_rate)
    windows: list[AudioSegment] = []
    start = 0
    while start < audio.sample_count:
        end = _window_end(audio, start, window_samples)
        windows.append(
            AudioSegment(
                data=audio.data[start:end],
                sample_rate=audio.sample_rate,
            ),
        )
        start = end
    return windows


def _window_end(audio: AudioSegment, start: int, window_samples: int) -> int:
    limit = start + window_samples
    if limit >= audio.sample_count:
        return audio.sample_count
    return _quietest_split_point(audio, start, limit)


def _quietest_split_point(audio: AudioSegment, start: int, limit: int) -> int:
    """Index of the quietest frame boundary near *limit*, else *limit*.

    Cutting on the loudest part of a syllable mangles a word at every
    seam, so the split moves to the lowest-energy frame in the trailing
    search slice. The returned index always advances past *start*, so a
    caller looping on it terminates.
    """
    frame = int(_BOUNDARY_FRAME_SECONDS * audio.sample_rate)
    if frame < 1:
        return limit
    search_samples = int(_BOUNDARY_SEARCH_SECONDS * audio.sample_rate)
    search_start = max(start + frame, limit - search_samples)
    frame_count = (limit - search_start) // frame
    if frame_count < 2:
        return limit
    frames = audio.data[search_start:search_start + frame_count * frame]
    energy = np.abs(frames.reshape(frame_count, frame)).mean(axis=1)
    # The last of the quietest frames, not the first: on a silent gap
    # that keeps the whole gap with this window so the next one opens on
    # speech, and on flat audio it splits as late as possible.
    quietest = int(np.flatnonzero(energy == energy.min())[-1])
    return search_start + (quietest + 1) * frame


def _decode_budget(model: WhisperForConditionalGeneration, prefix_length: int) -> int:
    """How many tokens generation may still emit after the prefix."""
    positions = getattr(model.config, "max_target_positions", None)
    if not isinstance(positions, int) or positions <= prefix_length + 1:
        return _FALLBACK_MAX_NEW_TOKENS
    return positions - prefix_length - 1


def resolve_crisperwhisper_mode(value: ConfigValue) -> CrisperWhisperMode:
    """Validate the persisted decoder-prefix mode at the engine boundary."""
    if value not in _VALID_MODES:
        raise ConfigurationError(
            "transcription.crisperwhisper_mode must be "
            "'intended' or 'verbatim'"
        )
    return cast(CrisperWhisperMode, value)


class CrisperWhisperAdapter(TranscriptionEngine):
    """Whisper adapter with Nyra's intended/verbatim decoder prefixes.

    The model and processor are loaded once from the app-managed Hugging Face
    cache and reused across utterances. No CrisperWhisper package, CT2, CUDA
    graph, or attention-capture path is involved.

    *revision* is the audited commit from ``ModelInfo.revision``, the same
    one the downloader fetched. The licence text this build gates on was
    audited at that commit, so loading any other one would show terms the
    weights no longer match.
    """

    def __init__(
        self,
        repo_id: str,
        revision: str,
        device: str,
        model_dir: Path,
        language: str,
        mode: CrisperWhisperMode = "intended",
    ) -> None:
        self._repo_id = repo_id
        self._revision = revision
        self._device = device
        self._model_dir = model_dir
        self._language = language
        self._mode = resolve_crisperwhisper_mode(mode)
        self._model: WhisperForConditionalGeneration | None = None
        self._processor: WhisperProcessor | None = None
        self._decoder_prefix: list[int] | None = None
        self._dtype: object | None = None
        self._torch_module: ModuleType | None = None

    def load_model(self) -> None:
        self._log_phase("load_model", f"start device={self._device}")
        started_at = time.monotonic()
        torch_module, model_type, processor_type = self._import_backend()
        dtype = self._dtype_for(torch_module, self._device)
        cache_dir = str(self._model_dir)

        processor = processor_type.from_pretrained(
            self._repo_id,
            revision=self._revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        model = model_type.from_pretrained(
            self._repo_id,
            revision=self._revision,
            cache_dir=cache_dir,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        model.to(self._device)
        model.eval()

        # decoder_input_ids below owns the complete prompt. Clearing model
        # defaults prevents Transformers from injecting a second language/task
        # prefix. Nyra also disables first-step blank suppression so silence is
        # allowed to produce an empty transcript.
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None
        model.generation_config.begin_suppress_tokens = None
        model.config.begin_suppress_tokens = None

        decoder_prefix = self._build_decoder_prefix(processor, model)
        self._processor = processor
        self._model = model
        self._dtype = dtype
        self._torch_module = torch_module
        self._decoder_prefix = decoder_prefix
        self._log_timed_phase("load_model", "complete", started_at)

    def is_model_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        self._raise_if_not_loaded()
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
        model = self._model
        processor = self._processor
        prefix = self._decoder_prefix
        dtype = self._dtype
        torch_module = self._torch_module
        if (
            model is None
            or processor is None
            or prefix is None
            or dtype is None
            or torch_module is None
        ):
            raise TranscriptionError("Model not loaded")

        processed = processor(
            audio.data,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
        )
        input_features = processed.input_features.to(
            device=self._device,
            dtype=dtype,
        )
        decoder_input_ids = torch_module.tensor(
            [prefix],
            device=self._device,
            dtype=torch_module.long,
        )
        suppress_tokens = self._resolved_suppress_tokens(model)
        with torch_module.inference_mode():
            generated = model.generate(
                input_features,
                language=self._language,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=_decode_budget(model, len(prefix)),
                num_beams=1,
                do_sample=False,
                use_cache=True,
                suppress_tokens=suppress_tokens,
            )

        token_ids = [int(token_id) for token_id in generated[0].tolist()]
        if token_ids[:len(prefix)] == prefix:
            token_ids = token_ids[len(prefix):]
        decoded = processor.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )
        return self._strip_prompt_artifacts(decoded)

    def _build_decoder_prefix(
        self,
        processor: WhisperProcessor,
        model: WhisperForConditionalGeneration,
    ) -> list[int]:
        # Nyra trained on five adjacent atomic mode tags followed by the
        # standard Whisper SOT/language/transcribe/no-timestamps sequence.
        tags = "".join(
            f"[{self._mode}_{index}]"
            for index in range(1, _MODE_TAG_COUNT + 1)
        )
        mode_ids = processor.tokenizer.encode(
            tags,
            add_special_tokens=False,
        )
        if len(mode_ids) != _MODE_TAG_COUNT:
            raise TranscriptionError(
                f"Model tokenizer does not contain atomic {self._mode} tags"
            )

        decoder_start_token_id = model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            raise TranscriptionError("Model has no decoder start token")
        forced_ids = processor.get_decoder_prompt_ids(
            task="transcribe",
            language=self._language,
            no_timestamps=True,
        )
        whisper_prefix = [int(decoder_start_token_id)] + [
            int(token_id)
            for _, token_id in sorted(forced_ids, key=lambda item: item[0])
        ]
        return [int(token_id) for token_id in mode_ids] + whisper_prefix

    @staticmethod
    def _resolved_suppress_tokens(
        model: WhisperForConditionalGeneration,
    ) -> list[int]:
        raw = model.generation_config.suppress_tokens
        if not raw:
            return []
        return [int(token_id) for token_id in raw if int(token_id) >= 0]

    @staticmethod
    def _strip_prompt_artifacts(text: str) -> str:
        text = re.sub(r"\[(?:verbatim|intended)_\d+\]", "", text)
        return " ".join(text.split()).strip()

    def _raise_if_not_loaded(self) -> None:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")

    def dispose(self) -> None:
        if not self.is_model_loaded():
            return
        self._log_phase("dispose", f"start device={self._device}")
        self._model = None
        self._processor = None
        self._decoder_prefix = None
        self._dtype = None
        self._torch_module = None
        self._log_phase("dispose", "refs nulled")
        if self._device != _CPU_DEVICE:
            import torch
            torch.cuda.empty_cache()
        self._log_phase("dispose", "cache released")

    @staticmethod
    def _dtype_for(torch_module: ModuleType, device: str) -> object:
        if device == _CPU_DEVICE:
            return torch_module.float32
        return torch_module.bfloat16

    @staticmethod
    def _import_backend() -> tuple[
        ModuleType,
        type[WhisperForConditionalGeneration],
        type[WhisperProcessor],
    ]:
        try:
            import torch
        except ImportError as exc:
            raise make_engine_dependency_error(
                package_name="torch",
                install_hint="PyTorch not installed: pip install torch",
                underlying=exc,
            ) from exc
        try:
            from transformers import (
                WhisperForConditionalGeneration,
                WhisperProcessor,
            )
        except ImportError as exc:
            raise make_engine_dependency_error(
                package_name="transformers",
                install_hint=(
                    "Transformers not installed: pip install "
                    "'transformers>=5.16.0,<6'"
                ),
                underlying=exc,
            ) from exc
        return torch, WhisperForConditionalGeneration, WhisperProcessor

    def _log_phase(self, operation: str, message: str) -> None:
        _logger.info(
            "[%s %s] %s | %s",
            operation,
            self._repo_id,
            message,
            vram_snapshot(),
        )

    def _log_timed_phase(
        self,
        operation: str,
        message: str,
        started_at: float,
    ) -> None:
        elapsed = time.monotonic() - started_at
        _logger.info(
            "[%s %s] %s in %.2fs | %s",
            operation,
            self._repo_id,
            message,
            elapsed,
            vram_snapshot(),
        )
