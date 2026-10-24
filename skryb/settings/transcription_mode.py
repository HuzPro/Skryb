from enum import Enum


_DISPLAY_LABELS: dict[str, str] = {
    "batch": "Batch",
    "streaming": "Streaming",
}


class TranscriptionMode(Enum):
    """Whether transcription runs after recording (batch) or live (streaming).

    Enum values are the strings persisted under ``transcription.mode`` in
    ``config.json``; changing a value is a persisted-format break.
    """

    BATCH = "batch"
    STREAMING = "streaming"

    @classmethod
    def migrate(cls, raw: str | None) -> "TranscriptionMode":
        """Map a persisted string to its mode, defaulting to ``BATCH``."""
        if raw is None:
            return cls.BATCH
        try:
            return cls(raw)
        except ValueError:
            return cls.BATCH

    @property
    def display_label(self) -> str:
        return _DISPLAY_LABELS[self.value]
