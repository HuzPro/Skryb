import pytest

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.transcription_mode import TranscriptionMode


def _make_draft(**overrides) -> SettingsDraft:
    base = dict(
        model_size="base",
        language="en",
        device="auto",
        engine="auto",
        hotkey_combination="ctrl+alt+m",
        hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE,
    )
    base.update(overrides)
    return SettingsDraft(**base)


class TestSettingsDraftConstruction:
    def test_should_hold_all_seven_fields(self):
        draft = _make_draft()

        assert draft.model_size == "base"
        assert draft.language == "en"
        assert draft.device == "auto"
        assert draft.engine == "auto"
        assert draft.hotkey_combination == "ctrl+alt+m"
        assert draft.hotkey_mode == "toggle"
        assert draft.output_mode is OutputMode.DIRECT_PASTE

    def test_should_be_immutable(self):
        draft = _make_draft()

        with pytest.raises(Exception):
            draft.model_size = "tiny"  # type: ignore[misc]


class TestChangeSetFromDrafts:
    def test_should_report_no_changes_when_drafts_equal(self):
        original = _make_draft()
        updated = _make_draft()

        changes = ChangeSet.between(original, updated)

        assert changes.engine_reload_required is False
        assert changes.hotkey_changed is False
        assert changes.output_changed is False

    def test_should_flag_engine_reload_required_when_model_size_differs(self):
        original = _make_draft(model_size="base")
        updated = _make_draft(model_size="small")

        changes = ChangeSet.between(original, updated)

        assert changes.engine_reload_required is True
        assert changes.hotkey_changed is False
        assert changes.output_changed is False

    def test_should_flag_engine_reload_required_when_device_differs(self):
        original = _make_draft(device="auto")
        updated = _make_draft(device="cuda")

        changes = ChangeSet.between(original, updated)

        assert changes.engine_reload_required is True

    def test_should_flag_engine_reload_required_when_language_differs(self):
        original = _make_draft(language="en")
        updated = _make_draft(language="fr")

        changes = ChangeSet.between(original, updated)

        assert changes.engine_reload_required is True
        assert changes.hotkey_changed is False
        assert changes.output_changed is False

    def test_should_flag_engine_reload_required_when_engine_differs(self):
        original = _make_draft(engine="auto")
        updated = _make_draft(engine="nano_parakeet")

        changes = ChangeSet.between(original, updated)

        assert changes.engine_reload_required is True
        assert changes.hotkey_changed is False
        assert changes.output_changed is False

    def test_should_flag_hotkey_change_when_combination_differs(self):
        original = _make_draft(hotkey_combination="ctrl+alt+m")
        updated = _make_draft(hotkey_combination="f12")

        changes = ChangeSet.between(original, updated)

        assert changes.hotkey_changed is True
        assert changes.engine_reload_required is False

    def test_should_flag_hotkey_change_when_mode_differs(self):
        original = _make_draft(hotkey_mode="toggle")
        updated = _make_draft(hotkey_mode="push_to_talk")

        changes = ChangeSet.between(original, updated)

        assert changes.hotkey_changed is True

    def test_should_flag_engine_reload_required_when_transcription_mode_differs(self):
        original = _make_draft(transcription_mode=TranscriptionMode.BATCH)
        updated = _make_draft(transcription_mode=TranscriptionMode.STREAMING)

        changes = ChangeSet.between(original, updated)

        assert changes.engine_reload_required is True
        assert changes.mode_changed is True

    def test_should_flag_output_change_when_mode_differs(self):
        original = _make_draft(output_mode=OutputMode.DIRECT_PASTE)
        updated = _make_draft(output_mode=OutputMode.CLIPBOARD_ONLY)

        changes = ChangeSet.between(original, updated)

        assert changes.output_changed is True

    def test_should_have_any_changed_helper(self):
        no_change = ChangeSet.between(_make_draft(), _make_draft())
        with_change = ChangeSet.between(_make_draft(), _make_draft(model_size="tiny"))

        assert no_change.any_changed is False
        assert with_change.any_changed is True
