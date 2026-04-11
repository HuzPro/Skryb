from pathlib import Path
from unittest.mock import MagicMock

from voicetotext.downloads.hf_repo_resolver import ResolvedFile
from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
from voicetotext.transcription.model_info import TRANSFORMERS_WHISPER_MODELS
from voicetotext.transcription.model_manager import ModelManager


def _make_manager(
    tmp_path: Path,
    download_manager=None,
    resolver=None,
    file_downloader=None,
    models=None,
) -> HuggingFaceModelManager:
    return HuggingFaceModelManager(
        models_dir=tmp_path,
        download_manager=download_manager or MagicMock(),
        resolver=resolver or MagicMock(),
        file_downloader=file_downloader or MagicMock(),
        models=models,
    )


class TestHuggingFaceModelManagerListModels:
    def test_should_list_all_available_models(self, tmp_path):
        manager = _make_manager(tmp_path)

        models = manager.list_models()

        assert len(models) == len(TRANSFORMERS_WHISPER_MODELS)

    def test_should_include_model_names(self, tmp_path):
        manager = _make_manager(tmp_path)

        names = {m.name for m in manager.list_models()}

        assert "base" in names
        assert "small" in names


class TestHuggingFaceModelManagerIsInstalled:
    def test_should_return_false_when_model_not_downloaded(self, tmp_path):
        manager = _make_manager(tmp_path)

        assert manager.is_installed("base") is False

    def test_should_return_true_when_snapshot_has_config_and_weights(self, tmp_path):
        snapshot = tmp_path / "models--openai--whisper-base" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").touch()
        (snapshot / "model.safetensors").touch()
        manager = _make_manager(tmp_path)

        assert manager.is_installed("base") is True

    def test_should_return_false_for_empty_model_directory(self, tmp_path):
        (tmp_path / "models--openai--whisper-base").mkdir()
        manager = _make_manager(tmp_path)

        assert manager.is_installed("base") is False

    def test_should_return_false_when_weights_file_missing(self, tmp_path):
        snapshot = (
            tmp_path / "models--openai--whisper-base" / "snapshots" / "abc123"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").touch()
        (snapshot / "tokenizer.json").touch()
        manager = _make_manager(tmp_path)

        assert manager.is_installed("base") is False

    def test_should_return_true_when_safetensors_present(self, tmp_path):
        snapshot = (
            tmp_path / "models--openai--whisper-base" / "snapshots" / "abc123"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").touch()
        (snapshot / "model.safetensors").touch()
        manager = _make_manager(tmp_path)

        assert manager.is_installed("base") is True


class TestHuggingFaceModelManagerGetModelPath:
    def test_should_return_none_when_not_installed(self, tmp_path):
        manager = _make_manager(tmp_path)

        assert manager.get_model_path("base") is None

    def test_should_return_path_when_installed(self, tmp_path):
        model_dir = tmp_path / "models--openai--whisper-base"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").touch()
        (snapshot / "model.safetensors").touch()
        manager = _make_manager(tmp_path)

        assert manager.get_model_path("base") == model_dir


class TestHuggingFaceModelManagerDownload:
    def test_should_resolve_repo_and_enqueue_job_on_manager(self, tmp_path):
        resolver = MagicMock()
        resolver.resolve.return_value = (
            "sha123",
            [
                ResolvedFile(
                    filename="config.json",
                    size=100,
                    url="https://huggingface.co/openai/whisper-base/resolve/sha123/config.json",
                ),
            ],
        )
        download_manager = MagicMock()
        manager = _make_manager(
            tmp_path, download_manager=download_manager, resolver=resolver
        )

        manager.download("base")

        resolver.resolve.assert_called_once_with("openai/whisper-base")
        download_manager.enqueue.assert_called_once()
        job = download_manager.enqueue.call_args[0][0]
        assert job.job_id == "base"
        assert job.repo_id == "openai/whisper-base"
        assert job.commit_sha == "sha123"

    def test_should_return_handle_from_download_manager(self, tmp_path):
        resolver = MagicMock()
        resolver.resolve.return_value = ("s", [])
        expected_handle = MagicMock()
        download_manager = MagicMock()
        download_manager.enqueue.return_value = expected_handle
        manager = _make_manager(
            tmp_path, download_manager=download_manager, resolver=resolver
        )

        handle = manager.download("base")

        assert handle is expected_handle


class TestHuggingFaceModelManagerDelete:
    def test_should_remove_model_directory(self, tmp_path):
        model_dir = tmp_path / "models--openai--whisper-base"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").touch()
        manager = _make_manager(tmp_path)

        manager.delete("base")

        assert not model_dir.exists()

    def test_should_not_raise_when_deleting_nonexistent_model(self, tmp_path):
        manager = _make_manager(tmp_path)

        manager.delete("base")


class TestHuggingFaceModelManagerIsAModelManager:
    def test_should_implement_model_manager_interface(self, tmp_path):
        manager = _make_manager(tmp_path)

        assert isinstance(manager, ModelManager)


class TestTransformersWhisperModelLineup:
    def test_lineup_contains_large_v3_turbo(self):
        from voicetotext.transcription.model_info import TRANSFORMERS_WHISPER_MODELS

        assert "large-v3-turbo" in TRANSFORMERS_WHISPER_MODELS
        info = TRANSFORMERS_WHISPER_MODELS["large-v3-turbo"]
        assert info.repo_id == "openai/whisper-large-v3-turbo"
        assert info.size_mb == 1620

    def test_lineup_does_not_contain_medium(self):
        from voicetotext.transcription.model_info import TRANSFORMERS_WHISPER_MODELS

        assert "medium" not in TRANSFORMERS_WHISPER_MODELS

    def test_lineup_uses_only_official_openai_repos(self):
        from voicetotext.transcription.model_info import TRANSFORMERS_WHISPER_MODELS

        for info in TRANSFORMERS_WHISPER_MODELS.values():
            assert info.repo_id.startswith("openai/whisper-")
