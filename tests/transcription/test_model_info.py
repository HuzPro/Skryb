from voicetotext.transcription.model_info import (
    TRANSFORMERS_WHISPER_MODELS,
    ModelInfo,
)


class TestModelInfoEntity:
    def test_should_store_model_attributes(self):
        info = ModelInfo(name="base", size_mb=142, description="Good balance")

        assert info.name == "base"
        assert info.size_mb == 142
        assert info.description == "Good balance"

    def test_should_format_size_in_mb(self):
        info = ModelInfo(name="small", size_mb=466, description="test")

        assert info.display_size == "466 MB"

    def test_should_format_size_in_gb_for_large_models(self):
        info = ModelInfo(name="medium", size_mb=1500, description="test")

        assert info.display_size == "1.5 GB"

    def test_should_store_explicit_repo_id(self):
        info = ModelInfo(
            name="base", size_mb=290, description="test",
            repo_id="openai/whisper-base",
        )

        assert info.repo_id == "openai/whisper-base"

    def test_repo_id_defaults_to_empty(self):
        info = ModelInfo(name="base", size_mb=142, description="test")

        assert info.repo_id == ""


class TestTransformersWhisperModels:
    def test_should_contain_all_standard_models(self):
        expected = {"tiny", "base", "small", "large-v3-turbo", "large-v3"}

        assert set(TRANSFORMERS_WHISPER_MODELS.keys()) == expected

    def test_all_models_should_have_repo_ids(self):
        for model in TRANSFORMERS_WHISPER_MODELS.values():
            assert model.repo_id.startswith("openai/whisper-")


