from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult


class FakeTranscriptionEngine(TranscriptionEngine):
    """In-memory TranscriptionEngine.

    Pre-program the result via :meth:`will_return` or the failure via
    :meth:`will_fail`.  Every call is recorded in :attr:`transcribe_calls`.
    """

    def __init__(self) -> None:
        self.transcribe_calls: list[AudioSegment] = []
        self.load_model_calls: int = 0
        self.dispose_called: bool = False
        self._model_loaded: bool = False
        self._next_result: TranscriptionResult | None = None
        self._error: Exception | None = None

    def will_return(self, result: TranscriptionResult) -> None:
        self._next_result = result
        self._error = None

    def will_fail(self, error: Exception) -> None:
        self._error = error

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        self.transcribe_calls.append(audio)
        if self._error is not None:
            raise self._error
        if self._next_result is None:
            raise AssertionError(
                "FakeTranscriptionEngine.transcribe called without a "
                "programmed result - call .will_return(...) first."
            )
        return self._next_result

    def load_model(self) -> None:
        self.load_model_calls += 1
        self._model_loaded = True

    def is_model_loaded(self) -> bool:
        return self._model_loaded

    def dispose(self) -> None:
        self.dispose_called = True
        self._model_loaded = False
