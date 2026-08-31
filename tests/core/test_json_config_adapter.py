import json


from voicetotext.core.config import ConfigurationStore
from voicetotext.core.json_config_adapter import JsonConfigAdapter


class TestJsonConfigAdapterDefaults:
    def test_should_return_default_for_unknown_key(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        assert config.get("nonexistent", "fallback") == "fallback"

    def test_should_return_none_when_no_default_given(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        assert config.get("nonexistent") is None

    def test_should_provide_default_config_on_first_load(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        assert config.get("hotkey.mode") == "toggle"
        assert config.get("hotkey.combination") == "ctrl+alt+m"
        assert config.get("output.mode") == "direct_paste"
        assert config.get("transcription.model_size") is None
        assert config.get("overlay.enabled") is True
        assert config.get("transcription.device") == "auto"
        assert config.get("audio.sample_rate") == 16000


class TestJsonConfigAdapterDotNotation:
    def test_should_get_nested_value_with_dot_notation(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        assert config.get("transcription.beam_size") == 5

    def test_should_set_nested_value_with_dot_notation(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        config.set("transcription.model_size", "nemotron-speech-streaming-en-0.6b")

        assert config.get("transcription.model_size") == "nemotron-speech-streaming-en-0.6b"

    def test_should_get_top_level_section_as_dict(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        hotkey = config.get("hotkey")

        assert isinstance(hotkey, dict)
        assert hotkey["mode"] == "toggle"


class TestJsonConfigAdapterPersistence:
    def test_should_distinguish_missing_profile_default_from_persisted_model(
        self, tmp_path,
    ):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)

        assert config.has_persisted_value("transcription.model_size") is False

        config.load()
        config.set("transcription.model_size", "parakeet-tdt-0.6b-v3")
        config.save()

        assert config.has_persisted_value("transcription.model_size") is True

    def test_should_persist_changes_to_file(self, tmp_path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        config.set("hotkey.mode", "push_to_talk")

        config.save()

        raw = json.loads(path.read_text())
        assert raw["hotkey"]["mode"] == "push_to_talk"

    def test_should_reload_persisted_changes(self, tmp_path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        config.set("output.mode", "clipboard_only")
        config.save()

        config2 = JsonConfigAdapter(path)
        config2.load()

        assert config2.get("output.mode") == "clipboard_only"

    def test_should_create_file_with_defaults_on_first_load(self, tmp_path):
        path = tmp_path / "config.json"
        config = JsonConfigAdapter(path)
        config.load()
        config.save()

        assert path.exists()
        raw = json.loads(path.read_text())
        assert "hotkey" in raw
        assert "transcription" in raw


class TestJsonConfigAdapterGetAll:
    def test_should_return_entire_config(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")
        config.load()

        all_config = config.get_all()

        assert "hotkey" in all_config
        assert "transcription" in all_config
        assert "output" in all_config
        assert "audio" in all_config


class TestJsonConfigAdapterIsAConfigurationStore:
    def test_should_implement_configuration_store_interface(self, tmp_path):
        config = JsonConfigAdapter(tmp_path / "config.json")

        assert isinstance(config, ConfigurationStore)
