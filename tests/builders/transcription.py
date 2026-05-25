from voicetotext.transcription.result import TranscriptionResult


class TranscriptionResultBuilder:
    """Fluent builder for :class:`TranscriptionResult`.

    Defaults to a typical "hello world" English result with high
    confidence.  ``.empty()`` produces a result with no text, which
    represents "no speech detected".
    """

    def __init__(self) -> None:
        self._text: str = "hello world"
        self._language: str = "en"
        self._language_probability: float = 0.95
        self._transcription_seconds: float = 1.0

    def with_text(self, text: str) -> "TranscriptionResultBuilder":
        self._text = text
        return self

    def empty(self) -> "TranscriptionResultBuilder":
        self._text = ""
        return self

    def in_language(
        self, language: str, probability: float = 0.95
    ) -> "TranscriptionResultBuilder":
        self._language = language
        self._language_probability = probability
        return self

    def lasting(self, seconds: float) -> "TranscriptionResultBuilder":
        self._transcription_seconds = seconds
        return self

    def build(self) -> TranscriptionResult:
        return TranscriptionResult(
            text=self._text,
            language=self._language,
            language_probability=self._language_probability,
            transcription_seconds=self._transcription_seconds,
        )


def a_transcription_result() -> TranscriptionResultBuilder:
    return TranscriptionResultBuilder()
