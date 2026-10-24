import json

from skryb.bootstrap import migrate_output_mode
from skryb.core.json_config_adapter import JsonConfigAdapter


class TestConfigRoundTrip:
    def test_should_persist_and_reload_all_settings(self, tmp_path):
        path = tmp_path / "config.json"

        config = JsonConfigAdapter(path)
        config.load()
        config.set("hotkey.mode", "push_to_talk")
        config.set("hotkey.combination", "f12")
        config.set("output.mode", "clipboard_only")
        config.set("transcription.model_size", "nemotron-speech-streaming-en-0.6b")
        config.set("transcription.device", "cuda")
        config.save()

        config2 = JsonConfigAdapter(path)
        config2.load()

        assert config2.get("hotkey.mode") == "push_to_talk"
        assert config2.get("hotkey.combination") == "f12"
        assert config2.get("output.mode") == "clipboard_only"
        assert config2.get("transcription.model_size") == "nemotron-speech-streaming-en-0.6b"
        assert config2.get("transcription.device") == "cuda"
        # Defaults should still be present for unmodified keys
        assert config2.get("transcription.beam_size") == 5
        assert config2.get("audio.sample_rate") == 16000

    def test_should_merge_new_defaults_into_existing_config(self, tmp_path):
        path = tmp_path / "config.json"

        # Write a partial config file (simulating an older version)
        partial = {"hotkey": {"mode": "toggle"}}
        path.write_text(json.dumps(partial))

        config = JsonConfigAdapter(path)
        config.load()

        # Old setting preserved
        assert config.get("hotkey.mode") == "toggle"
        # New defaults filled in
        assert config.get("transcription.model_size") is None
        assert config.get("output.mode") == "direct_paste"


class TestOutputModeMigration:
    def test_should_migrate_auto_paste_to_paste_and_clipboard(self, tmp_path):
        path = tmp_path / "config.json"
        old_config = {"output": {"mode": "auto_paste"}}
        path.write_text(json.dumps(old_config))

        config = JsonConfigAdapter(path)
        config.load()
        migrate_output_mode(config)

        assert config.get("output.mode") == "paste_and_clipboard"

    def test_should_preserve_direct_paste_through_migration(self, tmp_path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        migrate_output_mode(config)

        assert config.get("output.mode") == "direct_paste"
