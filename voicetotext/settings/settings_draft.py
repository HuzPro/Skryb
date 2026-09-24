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
    crisperwhisper_mode: str = "intended"
    capture_content: bool = False
    # "" is the Windows default microphone. Both are read at the start of
    # every recording, so neither needs a rebuild to take effect.
    input_device: str = ""
    boost_quiet_speech: bool = True


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

    @classmethod
    def between(cls, original: SettingsDraft, updated: SettingsDraft) -> "ChangeSet":
        # The intended/verbatim tags are baked into the decoder prefix at
        # load time, and verbatim also switches the output pipeline off,
        # so a mode change has to reload the engine and rebuild output.
        crisperwhisper_mode_changed = (
            original.crisperwhisper_mode != updated.crisperwhisper_mode
        )
        engine_reload_required = (
            original.model_size != updated.model_size
            or original.device != updated.device
            or original.language != updated.language
            or original.engine != updated.engine
            or original.transcription_mode != updated.transcription_mode
            or crisperwhisper_mode_changed
        )
        hotkey_changed = (
            original.hotkey_combination != updated.hotkey_combination
            or original.hotkey_mode != updated.hotkey_mode
        )
        mode_changed = original.transcription_mode != updated.transcription_mode
        # The pipeline runs in passthrough only while a CrisperWhisper model
        # is selected AND its mode is verbatim, so changing the model alone
        # can turn the dictionary, filler removal and number formatting back
        # on or off. That resolution lives in the composition root; rebuilding
        # output for any model change is what keeps this from having to hold a
        # second copy of it that can drift.
        output_changed = (
            original.output_mode != updated.output_mode
            or mode_changed
            or crisperwhisper_mode_changed
            or original.model_size != updated.model_size
        )
        return cls(
            engine_reload_required=engine_reload_required,
            hotkey_changed=hotkey_changed,
            output_changed=output_changed,
            mode_changed=mode_changed,
            diagnostics_changed=original.capture_content != updated.capture_content,
        )

    @property
    def any_changed(self) -> bool:
        return (
            self.engine_reload_required
            or self.hotkey_changed
            or self.output_changed
            or self.mode_changed
            or self.diagnostics_changed
        )
