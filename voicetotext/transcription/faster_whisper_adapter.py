import time

from faster_whisper import WhisperModel

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult


class FasterWhisperAdapter(TranscriptionEngine):

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        download_root: str | None = None,
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter
        self._download_root = download_root
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        kwargs = {"device": self._device, "compute_type": self._compute_type}
        if self._download_root:
            kwargs["download_root"] = self._download_root
        self._model = WhisperModel(self._model_size, **kwargs)

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if not self.is_model_loaded():
            raise TranscriptionError("Model not loaded")

        start = time.monotonic()
        segments, info = self._model.transcribe(
            audio.data,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=self._vad_filter,
        )

        text = "".join(seg.text for seg in segments).strip()
        duration = time.monotonic() - start

        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            duration_seconds=duration,
        )

    def dispose(self) -> None:
        self._model = None
