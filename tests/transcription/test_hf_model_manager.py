import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
from voicetotext.transcription.model_info import AVAILABLE_MODELS
from voicetotext.transcription.model_manager import ModelManager


class TestHuggingFaceModelManagerListModels:
    def test_should_list_all_available_models(self, tmp_path):
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        models = manager.list_models()

        assert len(models) == len(AVAILABLE_MODELS)

    def test_should_include_model_names(self, tmp_path):
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        models = manager.list_models()
        names = {m.name for m in models}

        assert "base" in names
        assert "small" in names


class TestHuggingFaceModelManagerIsInstalled:
    def test_should_return_false_when_model_not_downloaded(self, tmp_path):
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        assert manager.is_installed("base") is False

    def test_should_return_true_when_model_directory_exists(self, tmp_path):
        (tmp_path / "models--Systran--faster-whisper-base").mkdir()
        (tmp_path / "models--Systran--faster-whisper-base" / "model.bin").touch()
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        assert manager.is_installed("base") is True

    def test_should_return_false_for_empty_model_directory(self, tmp_path):
        (tmp_path / "models--Systran--faster-whisper-base").mkdir()
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        assert manager.is_installed("base") is False


class TestHuggingFaceModelManagerGetModelPath:
    def test_should_return_none_when_not_installed(self, tmp_path):
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        assert manager.get_model_path("base") is None

    def test_should_return_path_when_installed(self, tmp_path):
        model_dir = tmp_path / "models--Systran--faster-whisper-base"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        result = manager.get_model_path("base")

        assert result == model_dir


class TestHuggingFaceModelManagerDownload:
    @patch("voicetotext.transcription.hf_model_manager.snapshot_download")
    def test_should_call_snapshot_download_with_correct_repo(
        self, mock_download, tmp_path
    ):
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        manager.download("base", on_progress=lambda p: None)

        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args
        assert call_kwargs[0][0] == "Systran/faster-whisper-base"

    @patch("voicetotext.transcription.hf_model_manager.snapshot_download")
    def test_should_use_models_dir_as_cache(self, mock_download, tmp_path):
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        manager.download("small", on_progress=lambda p: None)

        call_kwargs = mock_download.call_args[1]
        assert str(tmp_path) in str(call_kwargs.get("cache_dir", ""))


class TestHuggingFaceModelManagerDelete:
    def test_should_remove_model_directory(self, tmp_path):
        model_dir = tmp_path / "models--Systran--faster-whisper-base"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        manager.delete("base")

        assert not model_dir.exists()

    def test_should_not_raise_when_deleting_nonexistent_model(self, tmp_path):
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        manager.delete("base")


class TestHuggingFaceModelManagerIsAModelManager:
    def test_should_implement_model_manager_interface(self, tmp_path):
        manager = HuggingFaceModelManager(models_dir=tmp_path)

        assert isinstance(manager, ModelManager)
