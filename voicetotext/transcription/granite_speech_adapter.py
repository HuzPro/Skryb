import logging
import time
from pathlib import Path

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.engine_diagnostics import vram_snapshot
from voicetotext.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)

_REQUIRED_SAMPLE_RATE = 16000
_MAX_NEW_TOKENS = 200
_PROMPT = (
    "<|audio|>transcribe the speech with proper punctuation "
    "and capitalization."
)


class GraniteSpeechAdapter(TranscriptionEngine):
    """Granite Speech 4.1 engine backed by transformers' Auto-classes.

    Granite is an autoregressive multimodal LLM — it does ASR via
    ``model.generate()`` over a chat-template prompt that contains the
    ``<|audio|>`` token. This is fundamentally different from Whisper's
    seq2seq forward pass, so it gets its own adapter rather than being
    multiplexed into ``TransformersWhisperAdapter``.
    """

    def __init__(
        self,
        repo_id: str,
        device: str,
        model_dir: Path,
        dtype: str = "bfloat16",
    ) -> None:
        self._repo_id = repo_id
        self._device = device
        self._model_dir = model_dir
        self._dtype_name = dtype
        self._processor: object | None = None
        self._model: object | None = None

    def load_model(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        cache_dir = str(self._model_dir)
        torch_dtype = getattr(torch, self._dtype_name)
        self._log_phase(
            "load_model",
            f"start device={self._device} dtype={self._dtype_name}",
        )
        started_at = time.monotonic()
        self._processor = AutoProcessor.from_pretrained(
            self._repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
            device_map=self._device,
            torch_dtype=torch_dtype,
        )
        self._log_timed_phase("load_model", "complete", started_at)

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
        return self._transcribe_loaded(audio)

    def _transcribe_loaded(self, audio: AudioSegment) -> TranscriptionResult:
        import torch

        wav = torch.from_numpy(audio.data).unsqueeze(0)
        chat = [{"role": "user", "content": _PROMPT}]
        prompt = self._processor.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            prompt,
            wav,
            device=self._device,
            return_tensors="pt",
        ).to(self._device)

        started_at = time.monotonic()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=_MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
        n_input_tokens = inputs["input_ids"].shape[-1]
        text = self._processor.tokenizer.batch_decode(
            out[:, n_input_tokens:],
            skip_special_tokens=True,
        )[0].strip()
        return TranscriptionResult(
            text=text,
            language="en",
            language_probability=0.0,
            duration_seconds=time.monotonic() - started_at,
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
