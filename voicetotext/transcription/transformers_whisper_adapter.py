import time

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult


class TransformersWhisperAdapter(TranscriptionEngine):
    """Whisper engine backed by HuggingFace *transformers* (pure PyTorch).

    Works on any GPU that PyTorch supports, including AMD via ROCm.
    Models are downloaded from HuggingFace Hub (``openai/whisper-*``).
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        language: str | None = None,
        beam_size: int = 5,
        download_root: str | None = None,
    ):
        self._model_size = model_size
        self._device = device
        self._language = language
        self._beam_size = beam_size
        self._download_root = download_root
        self._model = None
        self._processor = None

    def load_model(self) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        repo_id = f"openai/whisper-{self._model_size}"
        kwargs = {}
        if self._download_root:
            kwargs["cache_dir"] = self._download_root

        self._processor = WhisperProcessor.from_pretrained(repo_id, **kwargs)
        self._model = WhisperForConditionalGeneration.from_pretrained(repo_id, **kwargs)

        if self._device != "cpu":
            import torch

            self._model = self._model.to(self._device)
            if torch.cuda.is_available():
                self._model = self._model.half()

    def is_model_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")

        import torch

        start = time.monotonic()

        input_features = self._processor(
            audio.data,
            sampling_rate=audio.sample_rate,
            return_tensors="pt",
        ).input_features.to(self._device)

        if self._device != "cpu" and self._model.dtype != input_features.dtype:
            input_features = input_features.half()

        generate_kwargs: dict = {"num_beams": self._beam_size}
        if self._language:
            generate_kwargs["language"] = self._language
            generate_kwargs["task"] = "transcribe"

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features, **generate_kwargs
            )

        text = self._processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        duration = time.monotonic() - start

        language = self._language or "en"
        return TranscriptionResult(
            text=text,
            language=language,
            language_probability=0.0,
            duration_seconds=duration,
        )

    def dispose(self) -> None:
        self._model = None
        self._processor = None
