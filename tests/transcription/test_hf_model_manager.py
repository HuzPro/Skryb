from pathlib import Path
from unittest.mock import MagicMock

from voicetotext.downloads.hf_repo_resolver import ResolvedFile
from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
from voicetotext.transcription.model_info import WHISPER_MODELS
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

        assert len(models) == len(WHISPER_MODELS)

    def test_should_include_model_names(self, tmp_path):
        manager = _make_manager(tmp_path)

        names = {m.name for m in manager.list_models()}

        assert "base" in names
        assert "small" in names


class TestHuggingFaceModelManagerHasModel:
    def test_should_return_true_for_known_whisper_model(self, tmp_path):
        manager = _make_manager(tmp_path)

        assert manager.has_model("base") is True

    def test_should_return_false_for_unknown_model(self, tmp_path):
        manager = _make_manager(tmp_path)

        assert manager.has_model("does-not-exist") is False

    def test_should_only_recognise_models_in_its_own_dict(self, tmp_path):
        from voicetotext.transcription.model_info import PARAKEET_MODELS

        manager = _make_manager(tmp_path, models=PARAKEET_MODELS)

        assert manager.has_model("parakeet-tdt-0.6b-v3") is True
        assert manager.has_model("base") is False


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

    def test_should_return_true_when_sharded_safetensors_index_present(
        self, tmp_path,
    ):
        # Granite Speech 4.1-2b ships as sharded safetensors:
        # ``model-00001-of-00003.safetensors`` etc. plus
        # ``model.safetensors.index.json``. There is no single
        # ``model.safetensors`` file, so the install check must accept
        # the index-file layout as a complete model.
        from voicetotext.transcription.model_info import GRANITE_MODELS

        snapshot = (
            tmp_path
            / "models--ibm-granite--granite-speech-4.1-2b"
            / "snapshots"
            / "abc123"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").touch()
        (snapshot / "model.safetensors.index.json").touch()
        (snapshot / "model-00001-of-00003.safetensors").touch()
        (snapshot / "model-00002-of-00003.safetensors").touch()
        (snapshot / "model-00003-of-00003.safetensors").touch()
        manager = _make_manager(tmp_path, models=GRANITE_MODELS)

        assert manager.is_installed("granite-speech-4.1-2b") is True

    def test_should_return_true_for_nemo_only_snapshot(self, tmp_path):
        # Parakeet ships as a single .nemo file. nano-parakeet doesn't need
        # config.json or safetensors — just the .nemo. The install check
        # must recognise that as a complete model.
        from voicetotext.transcription.model_info import PARAKEET_MODELS

        snapshot = (
            tmp_path
            / "models--nvidia--parakeet-tdt-0.6b-v3"
            / "snapshots"
            / "abc123"
        )
        snapshot.mkdir(parents=True)
        (snapshot / "parakeet-tdt-0.6b-v3.nemo").touch()
        manager = _make_manager(tmp_path, models=PARAKEET_MODELS)

        assert manager.is_installed("parakeet-tdt-0.6b-v3") is True


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


class TestWhisperModelLineup:
    def test_lineup_contains_large_v3_turbo(self):
        from voicetotext.transcription.model_info import WHISPER_MODELS

        assert "large-v3-turbo" in WHISPER_MODELS
        info = WHISPER_MODELS["large-v3-turbo"]
        assert info.repo_id == "openai/whisper-large-v3-turbo"
        assert info.size_mb == 1620

    def test_lineup_does_not_contain_medium(self):
        from voicetotext.transcription.model_info import WHISPER_MODELS

        assert "medium" not in WHISPER_MODELS

    def test_lineup_uses_only_official_openai_repos(self):
        from voicetotext.transcription.model_info import WHISPER_MODELS

        for info in WHISPER_MODELS.values():
            assert info.repo_id.startswith("openai/whisper-")
