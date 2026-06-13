from dataclasses import dataclass

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.transcription_mode import TranscriptionMode


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
    hotkey_combination: str
    hotkey_mode: str
    output_mode: OutputMode
    transcription_mode: TranscriptionMode = TranscriptionMode.BATCH


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

    @classmethod
    def between(cls, original: SettingsDraft, updated: SettingsDraft) -> "ChangeSet":
        engine_reload_required = (
            original.model_size != updated.model_size
            or original.device != updated.device
            or original.language != updated.language
            or original.engine != updated.engine
            or original.transcription_mode != updated.transcription_mode
        )
        hotkey_changed = (
            original.hotkey_combination != updated.hotkey_combination
            or original.hotkey_mode != updated.hotkey_mode
        )
        output_changed = original.output_mode != updated.output_mode
        mode_changed = original.transcription_mode != updated.transcription_mode
        return cls(
            engine_reload_required=engine_reload_required,
            hotkey_changed=hotkey_changed,
            output_changed=output_changed,
            mode_changed=mode_changed,
        )

    @property
    def any_changed(self) -> bool:
        return (
            self.engine_reload_required
            or self.hotkey_changed
            or self.output_changed
            or self.mode_changed
        )
