from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.settings.transcription_mode import TranscriptionMode


class TestTranscriptionModeMigrate:
    def test_should_default_to_batch_for_none(self):
        assert TranscriptionMode.migrate(None) is TranscriptionMode.BATCH

    def test_should_parse_streaming(self):
        assert TranscriptionMode.migrate("streaming") is TranscriptionMode.STREAMING

    def test_should_parse_batch(self):
        assert TranscriptionMode.migrate("batch") is TranscriptionMode.BATCH

    def test_should_fall_back_to_batch_for_unknown(self):
        assert TranscriptionMode.migrate("nonsense") is TranscriptionMode.BATCH


class TestTranscriptionModeValues:
    def test_values_are_the_config_strings(self):
        assert TranscriptionMode.BATCH.value == "batch"
        assert TranscriptionMode.STREAMING.value == "streaming"

    def test_each_mode_has_a_display_label(self):
        for mode in TranscriptionMode:
            assert mode.display_label


class TestConfigDefault:
    def test_fresh_config_defaults_mode_to_batch(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        assert config.get("transcription.mode") == "batch"
