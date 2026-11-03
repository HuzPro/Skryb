import pytest

from skryb.hotkeys.bindings import Binding, BindingAction, BindingMode
from skryb.output.output_mode import OutputMode
from skryb.settings.settings_draft import ChangeSet, SettingsDraft
from skryb.settings.transcription_mode import TranscriptionMode


def _make_draft(**overrides) -> SettingsDraft:
    base = dict(
        model_size="parakeet-ultra",
        language="en",
        device="auto",
        engine="auto",
        output_mode=OutputMode.DIRECT_PASTE,
    )
    base.update(overrides)
    return SettingsDraft(**base)


class TestSettingsDraftConstruction:
    def test_should_hold_all_seven_fields(self):
        draft = _make_draft()

        assert draft.model_size == "parakeet-ultra"
        assert draft.language == "en"
        assert draft.device == "auto"
        assert draft.engine == "auto"
        assert draft.hotkey_bindings == (
            Binding("ctrl+alt+m", BindingMode.TOGGLE, BindingAction.DICTATE),
        )
        assert draft.double_tap_locks_hold is True
        assert draft.output_mode is OutputMode.DIRECT_PASTE

    def test_should_be_immutable(self):
        draft = _make_draft()

        with pytest.raises(Exception):
            draft.model_size = "test-model"  # type: ignore[misc]


class TestChangeSetFromDrafts:
    def test_should_report_no_changes_when_drafts_equal(self):
        original = _make_draft()
        updated = _make_draft()

        changes = ChangeSet.between(original, updated)

        assert changes.engine_reload_required is False
        assert changes.hotkey_changed is False
        assert changes.output_changed is False

    def test_should_reload_and_rebuild_output_when_model_size_differs(self):
        original = _make_draft(model_size="parakeet-ultra")
        updated = _make_draft(model_size="nemotron-speech-streaming-en-0.6b")

        changes = ChangeSet.between(original, updated)

        assert changes.engine_reload_required is True
        assert changes.hotkey_changed is False
        assert changes.output_changed is True

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

    def test_should_flag_hotkey_change_when_a_shortcut_is_added(self):
        original = _make_draft()
        updated = _make_draft(hotkey_bindings=original.hotkey_bindings + (
            Binding("mouse4", BindingMode.HOLD, BindingAction.DICTATE),
        ))

        changes = ChangeSet.between(original, updated)

        assert changes.hotkey_changed is True
        assert changes.engine_reload_required is False

    def test_should_flag_hotkey_change_when_a_mode_differs(self):
        original = _make_draft(hotkey_bindings=(
            Binding("f12", BindingMode.TOGGLE, BindingAction.DICTATE),
        ))
        updated = _make_draft(hotkey_bindings=(
            Binding("f12", BindingMode.HOLD, BindingAction.DICTATE),
        ))

        changes = ChangeSet.between(original, updated)

        assert changes.hotkey_changed is True

    def test_should_flag_hotkey_change_when_the_double_tap_lock_is_switched(self):
        changes = ChangeSet.between(
            _make_draft(double_tap_locks_hold=True),
            _make_draft(double_tap_locks_hold=False),
        )

        assert changes.hotkey_changed is True

    def test_should_flag_engine_reload_required_when_transcription_mode_differs(self):
        original = _make_draft(transcription_mode=TranscriptionMode.BATCH)
        updated = _make_draft(transcription_mode=TranscriptionMode.STREAMING)

        changes = ChangeSet.between(original, updated)

        assert changes.engine_reload_required is True
        assert changes.mode_changed is True
        assert changes.output_changed is True

    def test_should_flag_output_change_when_mode_differs(self):
        original = _make_draft(output_mode=OutputMode.DIRECT_PASTE)
        updated = _make_draft(output_mode=OutputMode.CLIPBOARD_ONLY)

        changes = ChangeSet.between(original, updated)

        assert changes.output_changed is True

    def test_should_have_any_changed_helper(self):
        no_change = ChangeSet.between(_make_draft(), _make_draft())
        with_change = ChangeSet.between(_make_draft(), _make_draft(model_size="test-model"))

        assert no_change.any_changed is False
        assert with_change.any_changed is True


class TestHearingQuestionsChanges:
    def test_turning_it_off_reloads_the_engine_that_applies_it(self) -> None:
        changes = ChangeSet.between(_make_draft(question_marks=True), _make_draft(question_marks=False))

        assert changes.engine_reload_required

    def test_leaving_it_alone_changes_nothing(self) -> None:
        draft = _make_draft(question_marks=False)

        assert not ChangeSet.between(draft, draft).any_changed
