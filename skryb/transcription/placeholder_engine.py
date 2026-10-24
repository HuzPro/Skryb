from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.result import TranscriptionResult


class PlaceholderEngine(TranscriptionEngine):
    """Inert engine returned while the real engine is being constructed.

    Bootstrap moves the real engine load off the main thread so the
    tray comes up immediately. The use case still needs a non-None
    ``TranscriptionEngine`` reference at construction time - this is
    that reference. The use case also blocks recording attempts while
    the state machine is in ``LOADING``, so ``transcribe`` should
    never actually be invoked; raising rather than returning empty
    output makes any future bug here loud instead of silent.
    """

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        raise TranscriptionError(
            "Transcription engine is still loading"
        )

    def load_model(self) -> None:
        return None

    def is_model_loaded(self) -> bool:
        return False

    def dispose(self) -> None:
        return None

    def warmup(self) -> None:
        # Override the base default which calls ``transcribe`` - that
        # would raise. The placeholder has nothing to warm up.
        return None
