from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    transcription_seconds: float

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()
