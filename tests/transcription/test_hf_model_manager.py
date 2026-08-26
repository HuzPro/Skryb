from pathlib import Path
from unittest.mock import MagicMock

from voicetotext.downloads.hf_repo_resolver import ResolvedFile
from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
from voicetotext.transcription.model_info import (
    DEFAULT_MODEL,
    PARAKEET_MODELS,
    ModelInfo,
)
from voicetotext.transcription.model_manager import ModelManager


def _make_manager(
    tmp_path: Path,
    *,
    download_manager: MagicMock | None = None,
    resolver: MagicMock | None = None,
    models: dict[str, ModelInfo] = PARAKEET_MODELS,
) -> HuggingFaceModelManager:
    return HuggingFaceModelManager(
        models_dir=tmp_path,
        download_manager=download_manager or MagicMock(),
        resolver=resolver or MagicMock(),
        file_downloader=MagicMock(),
        models=models,
    )


def _model_dir(root: Path) -> Path:
    return root / "models--nvidia--parakeet-tdt-0.6b-v3"


def _write_refs_main(model_dir: Path, sha: str = "abc123") -> None:
    refs_main = model_dir / "refs" / "main"
    refs_main.parent.mkdir(parents=True, exist_ok=True)
    refs_main.write_text(sha)


def _write_nemo_snapshot(root: Path, *, with_ref: bool = True) -> Path:
    model_dir = _model_dir(root)
    snapshot = model_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "parakeet.nemo").touch()
    if with_ref:
        _write_refs_main(model_dir)
    return model_dir


class TestHuggingFaceModelManagerCatalog:
    def test_should_list_supplied_models(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)

        assert [model.name for model in manager.list_models()] == [DEFAULT_MODEL]
        assert manager.has_model(DEFAULT_MODEL) is True
        assert manager.has_model("unknown") is False


class TestHuggingFaceModelManagerInstallation:
    def test_should_return_false_when_model_is_not_downloaded(
        self, tmp_path: Path,
    ) -> None:
        assert _make_manager(tmp_path).is_installed(DEFAULT_MODEL) is False

    def test_should_accept_nemo_snapshot(self, tmp_path: Path) -> None:
        _write_nemo_snapshot(tmp_path)

        assert _make_manager(tmp_path).is_installed(DEFAULT_MODEL) is True

    def test_should_reject_snapshot_without_refs_main(
        self, tmp_path: Path,
    ) -> None:
        _write_nemo_snapshot(tmp_path, with_ref=False)

        assert _make_manager(tmp_path).is_installed(DEFAULT_MODEL) is False

    def test_should_reject_ref_to_missing_snapshot(self, tmp_path: Path) -> None:
        model_dir = _model_dir(tmp_path)
        model_dir.mkdir(parents=True)
        _write_refs_main(model_dir, "missing")

        assert _make_manager(tmp_path).is_installed(DEFAULT_MODEL) is False

    def test_should_accept_sharded_safetensors_layout(
        self, tmp_path: Path,
    ) -> None:
        info = ModelInfo(
            name="example-sharded", size_mb=10, description="Example",
            repo_id="example/sharded", compatible_engines=("nano_parakeet",),
        )
        model_dir = tmp_path / "models--example--sharded"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").touch()
        (snapshot / "model.safetensors.index.json").touch()
        _write_refs_main(model_dir)

        manager = _make_manager(tmp_path, models={info.name: info})

        assert manager.is_installed(info.name) is True


class TestHuggingFaceModelManagerOperations:
    def test_get_model_path_should_reflect_installation(
        self, tmp_path: Path,
    ) -> None:
        assert _make_manager(tmp_path).get_model_path(DEFAULT_MODEL) is None
        model_dir = _write_nemo_snapshot(tmp_path)

        assert _make_manager(tmp_path).get_model_path(DEFAULT_MODEL) == model_dir

    def test_download_should_resolve_and_enqueue(self, tmp_path: Path) -> None:
        resolver = MagicMock()
        resolver.resolve.return_value = (
            "sha123",
            [
                ResolvedFile(
                    filename="model.nemo",
                    size=100,
                    url=(
                        "https://huggingface.co/nvidia/"
                        "parakeet-tdt-0.6b-v3/resolve/sha123/model.nemo"
                    ),
                ),
            ],
        )
        download_manager = MagicMock()
        expected_handle = MagicMock()
        download_manager.enqueue.return_value = expected_handle
        manager = _make_manager(
            tmp_path, resolver=resolver, download_manager=download_manager,
        )

        handle = manager.download(DEFAULT_MODEL)

        resolver.resolve.assert_called_once_with(
            "nvidia/parakeet-tdt-0.6b-v3",
        )
        job = download_manager.enqueue.call_args.args[0]
        assert job.job_id == DEFAULT_MODEL
        assert job.repo_id == "nvidia/parakeet-tdt-0.6b-v3"
        assert handle is expected_handle

    def test_delete_should_remove_model_directory(self, tmp_path: Path) -> None:
        model_dir = _write_nemo_snapshot(tmp_path)
        manager = _make_manager(tmp_path)

        manager.delete(DEFAULT_MODEL)

        assert model_dir.exists() is False

    def test_should_implement_model_manager_interface(self, tmp_path: Path) -> None:
        assert isinstance(_make_manager(tmp_path), ModelManager)
