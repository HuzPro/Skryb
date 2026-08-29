class VoiceToTextError(Exception):
    pass


class AudioRecorderError(VoiceToTextError):
    pass


class TranscriptionError(VoiceToTextError):
    pass


class EngineDependencyMissingError(TranscriptionError):
    """The Python package backing an engine is not installed.

    Raised by adapter ``load_model()`` implementations when the lazy
    backend import fails. The message must be actionable - it is
    surfaced verbatim to the user via the tray (e.g. ``"nano-parakeet
    not installed: pip install nano-parakeet"``).
    """
    pass


class EngineWarmupError(TranscriptionError):
    """Engine warmup raised an unrecoverable error.

    Distinct from ``EngineDependencyMissingError`` so callers can tell
    "the engine cannot run at all" from "warmup hit a transient backend
    quirk". The bootstrap warmup helper only converts unrecoverable
    underlying errors (``ImportError``, ``FileNotFoundError``,
    ``OSError``) into this exception; transient ``RuntimeError``\\ s
    stay swallowed so a backend quirk on silent audio doesn't crash
    startup.
    """
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


class LicenseNotAcceptedError(VoiceToTextError):
    """A gated model was acquired or used without a current acceptance.

    Raised by
    :meth:`~voicetotext.transcription.license_policy.LicensePolicy.assert_accepted`
    from every path that downloads, builds, or loads weights whose licence
    the user has not agreed to at the digest the app displays. The message
    is surfaced verbatim to the user, so it names the model and the terms.
    """
    pass


class HotkeyError(VoiceToTextError):
    pass
