class VoiceToTextError(Exception):
    pass


class AudioRecorderError(VoiceToTextError):
    pass


class TranscriptionError(VoiceToTextError):
    pass


class OutputError(VoiceToTextError):
    pass


class ConfigurationError(VoiceToTextError):
    pass


class HotkeyError(VoiceToTextError):
    pass
