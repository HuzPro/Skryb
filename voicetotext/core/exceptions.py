class VoiceToTextError(Exception):
    pass


class AudioRecorderError(VoiceToTextError):
    pass


class TranscriptionError(VoiceToTextError):
    pass


class EngineSwapError(VoiceToTextError):
    """Raised when a transcription engine hot-swap cannot be performed.

    Covers both rejected requests (swap already in progress, app not
    idle, first-run setup) and unrecoverable failures during the swap
    itself. Carries the operator-facing reason in the message.
    """
    pass


class OutputError(VoiceToTextError):
    pass


class ConfigurationError(VoiceToTextError):
    pass


class HotkeyError(VoiceToTextError):
    pass
