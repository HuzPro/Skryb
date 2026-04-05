from voicetotext.transcription.model_info import AVAILABLE_MODELS, ModelInfo


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

    def test_should_compute_huggingface_repo_id(self):
        info = ModelInfo(name="base", size_mb=142, description="test")

        assert info.repo_id == "Systran/faster-whisper-base"


class TestAvailableModels:
    def test_should_contain_all_standard_models(self):
        expected = {"tiny", "base", "small", "medium", "large-v3"}

        assert set(AVAILABLE_MODELS.keys()) == expected

    def test_should_have_correct_base_model_size(self):
        assert AVAILABLE_MODELS["base"].size_mb == 142

    def test_should_have_correct_tiny_model_size(self):
        assert AVAILABLE_MODELS["tiny"].size_mb == 75

    def test_all_models_should_have_repo_ids(self):
        for model in AVAILABLE_MODELS.values():
            assert model.repo_id.startswith("Systran/faster-whisper-")
