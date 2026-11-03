from dataclasses import dataclass

from skryb.hotkeys.bindings import DEFAULT_BINDINGS, Binding
from skryb.output.output_mode import OutputMode
from skryb.recording.preferences import RecordingPreferences
from skryb.settings.transcription_mode import TranscriptionMode


@dataclass(frozen=True)
class SettingsDraft:
    """Value object representing a pending set of settings changes.

    Carried from the tabs back to :class:`SettingsService.apply_draft` in
    one piece so the facade can persist atomically and compute a
    :class:`ChangeSet` without the UI needing to know which fields map to
    which config keys.
    """

    model_size: str
    language: str
    device: str
    engine: str
    output_mode: OutputMode
    transcription_mode: TranscriptionMode = TranscriptionMode.BATCH
    # Handed to the engine when it is built, like the speech style.
    question_marks: bool = True
    capture_content: bool = False
    # "" is the Windows default microphone. Both are read at the start of
    # every recording, so neither needs a rebuild to take effect.
    input_device: str = ""
    boost_quiet_speech: bool = True
    hotkey_bindings: tuple[Binding, ...] = DEFAULT_BINDINGS
    double_tap_locks_hold: bool = True
    # Read where it is used, at each recording or sound, so none of it
    # needs a rebuild either.
    recording: RecordingPreferences = RecordingPreferences()
    # 0 keeps history forever. Overrides are (process, method value) pairs,
    # read at every paste, so neither needs a rebuild.
    history_retention_days: int = 0
    paste_method_overrides: tuple[tuple[str, str], ...] = ()
    # The updater reads it before every hourly check.
    check_for_updates: bool = True
    # Read when a meeting starts, so none of them needs a rebuild.
    meetings_folder: str = ""
    meetings_record_microphone: bool = True
    meetings_language: str = "en"


@dataclass(frozen=True)
class ChangeSet:
    """Which groups of settings changed between two drafts.

    The facade publishes this back to the window so the save flow knows
    whether to trigger an engine hot-swap, rebuild the hotkey listener,
    or reconfigure output.
    """

    engine_reload_required: bool
    hotkey_changed: bool
    output_changed: bool
    mode_changed: bool = False
    diagnostics_changed: bool = False
    history_retention_changed: bool = False

    @classmethod
    def between(cls, original: SettingsDraft, updated: SettingsDraft) -> "ChangeSet":
        engine_reload_required = (
            original.model_size != updated.model_size
            or original.device != updated.device
            or original.language != updated.language
            or original.engine != updated.engine
            or original.transcription_mode != updated.transcription_mode
            or original.question_marks != updated.question_marks
        )
        hotkey_changed = (
            original.hotkey_bindings != updated.hotkey_bindings
            or original.double_tap_locks_hold != updated.double_tap_locks_hold
        )
        mode_changed = original.transcription_mode != updated.transcription_mode
        output_changed = (
            original.output_mode != updated.output_mode
            or mode_changed
            or original.model_size != updated.model_size
        )
        return cls(
            engine_reload_required=engine_reload_required,
            hotkey_changed=hotkey_changed,
            output_changed=output_changed,
            mode_changed=mode_changed,
            diagnostics_changed=original.capture_content != updated.capture_content,
            history_retention_changed=(
                original.history_retention_days != updated.history_retention_days
            ),
        )

    @property
    def any_changed(self) -> bool:
        return (
            self.engine_reload_required
            or self.hotkey_changed
            or self.output_changed
            or self.mode_changed
            or self.diagnostics_changed
            or self.history_retention_changed
        )
