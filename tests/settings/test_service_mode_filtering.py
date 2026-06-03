from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.transcription_mode import TranscriptionMode
from voicetotext.transcription.engine_factory import build_engine_registry
from voicetotext.transcription.model_info import ALL_MODELS

from tests.settings.test_service import _build_service


def _draft(mode: TranscriptionMode) -> SettingsDraft:
    return SettingsDraft(
        model_size="base", language="en", device="auto", engine="auto",
        hotkey_combination="ctrl+alt+m", hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE, transcription_mode=mode,
    )


class TestSettingsDraftMode:
    def test_should_default_transcription_mode_to_batch(self):
        draft = SettingsDraft(
            model_size="base", language="en", device="auto", engine="auto",
            hotkey_combination="x", hotkey_mode="toggle",
            output_mode=OutputMode.DIRECT_PASTE,
        )

        assert draft.transcription_mode is TranscriptionMode.BATCH

    def test_should_flag_mode_change_in_changeset(self):
        changes = ChangeSet.between(
            _draft(TranscriptionMode.BATCH), _draft(TranscriptionMode.STREAMING),
        )

        assert changes.mode_changed is True
        assert changes.any_changed is True


class TestServiceModeRoundTrip:
    def test_should_read_streaming_mode_from_config(self):
        service, *_ = _build_service(
            config_overrides={"transcription.mode": "streaming"},
        )

        assert service.current_draft().transcription_mode is (
            TranscriptionMode.STREAMING
        )

    def test_should_persist_mode_on_apply(self):
        service, config, *_ = _build_service()

        service.apply_draft(_draft(TranscriptionMode.STREAMING))

        assert config.get("transcription.mode") == "streaming"


class TestServiceModeFiltering:
    def test_should_list_all_models_in_batch_mode(self):
        service, *_ = _build_service(
            models=ALL_MODELS, registry=build_engine_registry(),
        )

        names = {view.name for view in service.list_models()}

        assert "base" in names
        assert "parakeet-tdt-0.6b-v3" in names

    def test_should_list_only_streaming_models_in_streaming_mode(self):
        service, *_ = _build_service(
            models=ALL_MODELS, registry=build_engine_registry(),
            config_overrides={"transcription.mode": "streaming"},
        )

        names = {view.name for view in service.list_models()}

        assert names == {"parakeet-tdt-0.6b-v3"}

    def test_should_report_streaming_capability_per_model(self):
        service, *_ = _build_service(registry=build_engine_registry())

        assert service.model_supports_streaming("parakeet-tdt-0.6b-v3") is True
        assert service.model_supports_streaming("base") is False
