import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.fakes.config import FakeConfig
from tests.fakes.licensing import AllowAllLicensePolicy
from skryb.core.exceptions import LicenseNotAcceptedError
from skryb.downloads.hf_repo_resolver import ResolvedFile, ResolvedRepo
from skryb.transcription.hf_model_manager import HuggingFaceModelManager
from skryb.transcription.license_policy import LicensePolicy
from skryb.transcription.model_info import (
    CRISPERWHISPER_MODELS,
    PARAKEET_MODELS,
    ModelInfo,
)
from skryb.transcription.model_manager import ModelManager


_PINNED_SHA = "1e28271b79edc97635783bea65abc89195a09ed3"
_LEGACY_SHA = "0000000000000000000000000000000000000000"

_PINNED_MODEL = ModelInfo(
    name="pinned-example", size_mb=10, description="Example",
    repo_id="example/pinned", revision=_PINNED_SHA,
    compatible_engines=("nano_parakeet",),
)

# Unpinned, so it tracks main, and saved as one .nemo file.
_NEMO_MODEL = ModelInfo(
    name="example-nemo", size_mb=10, description="Example",
    repo_id="example/nemo", compatible_engines=("nano_parakeet",),
)

_GATED_MODEL_NAME = "crisperwhisper-2.0-turbo"

_EXAMPLE_MODEL = ModelInfo(
    name="example-transformers", size_mb=10, description="Example",
    repo_id="example/transformers", compatible_engines=("nano_parakeet",),
)

_SHARD_NAMES = (
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)


def _make_manager(
    tmp_path: Path,
    *,
    download_manager: MagicMock | None = None,
    resolver: MagicMock | None = None,
    models: dict[str, ModelInfo] | None = None,
    license_policy: LicensePolicy | None = None,
) -> HuggingFaceModelManager:
    return HuggingFaceModelManager(
        models_dir=tmp_path,
        download_manager=download_manager or MagicMock(),
        resolver=resolver or MagicMock(),
        file_downloader=MagicMock(),
        models=models or {_NEMO_MODEL.name: _NEMO_MODEL},
        license_policy=license_policy or AllowAllLicensePolicy(),
    )


def _model_dir(root: Path) -> Path:
    return root / "models--example--nemo"


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


def _write_transformers_snapshot(root: Path, weight_files: dict[str, str]) -> None:
    model_dir = root / "models--example--transformers"
    snapshot = model_dir / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").touch()
    for filename, content in weight_files.items():
        (snapshot / filename).write_text(content)
    _write_refs_main(model_dir)


def _shard_index_json(shard_names: tuple[str, ...]) -> str:
    return json.dumps({
        "metadata": {"total_size": 8},
        "weight_map": {
            f"encoder.layer.{position}.weight": name
            for position, name in enumerate(shard_names)
        },
    })


def _write_sharded_snapshot(
    root: Path, *, shards_on_disk: tuple[str, ...], index_text: str = "",
) -> None:
    weight_files = {
        "model.safetensors.index.json": index_text
        or _shard_index_json(_SHARD_NAMES),
    }
    weight_files.update({shard_name: "" for shard_name in shards_on_disk})
    _write_transformers_snapshot(root, weight_files)


def _example_model_is_installed(tmp_path: Path) -> bool:
    manager = _make_manager(
        tmp_path, models={_EXAMPLE_MODEL.name: _EXAMPLE_MODEL},
    )
    return manager.is_installed(_EXAMPLE_MODEL.name)


def _pinned_manager(tmp_path: Path) -> HuggingFaceModelManager:
    return _make_manager(tmp_path, models={_PINNED_MODEL.name: _PINNED_MODEL})


def _write_pinned_snapshot(
    root: Path, *, snapshot_sha: str = _PINNED_SHA, refs_sha: str = _PINNED_SHA,
) -> Path:
    model_dir = root / "models--example--pinned"
    snapshot = model_dir / "snapshots" / snapshot_sha
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").touch()
    (snapshot / "model.safetensors").touch()
    if refs_sha:
        _write_refs_main(model_dir, refs_sha)
    return model_dir


class TestHuggingFaceModelManagerCatalog:
    def test_should_list_supplied_models(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path, models=PARAKEET_MODELS)

        assert [model.name for model in manager.list_models()] == [
            "parakeet-ultra", "underdog-bark-3.0", "underdog-bark-3.0-plus", "underdog-bark-3.0-clean", "underdog-bark-3.0-pro",
        ]
        assert manager.has_model("parakeet-ultra") is True
        assert manager.has_model("unknown") is False


class TestHuggingFaceModelManagerInstallation:
    def test_should_return_false_when_model_is_not_downloaded(
        self, tmp_path: Path,
    ) -> None:
        assert _make_manager(tmp_path).is_installed(_NEMO_MODEL.name) is False

    def test_should_accept_nemo_snapshot(self, tmp_path: Path) -> None:
        _write_nemo_snapshot(tmp_path)

        assert _make_manager(tmp_path).is_installed(_NEMO_MODEL.name) is True

    def test_should_reject_snapshot_without_refs_main(
        self, tmp_path: Path,
    ) -> None:
        _write_nemo_snapshot(tmp_path, with_ref=False)

        assert _make_manager(tmp_path).is_installed(_NEMO_MODEL.name) is False

    def test_should_reject_ref_to_missing_snapshot(self, tmp_path: Path) -> None:
        model_dir = _model_dir(tmp_path)
        model_dir.mkdir(parents=True)
        _write_refs_main(model_dir, "missing")

        assert _make_manager(tmp_path).is_installed(_NEMO_MODEL.name) is False

    def test_should_accept_single_file_safetensors_layout(
        self, tmp_path: Path,
    ) -> None:
        _write_transformers_snapshot(tmp_path, {"model.safetensors": ""})

        assert _example_model_is_installed(tmp_path) is True


class TestPackedArchiveInstallation:
    """Phonon-2 ships its whole model as one zstd tar beside a config.json."""

    def test_a_snapshot_holding_the_archive_is_installed(self, tmp_path: Path) -> None:
        _write_transformers_snapshot(tmp_path, {"phonon-2.bps.tar.zst": "zstd tar"})

        assert _example_model_is_installed(tmp_path) is True

    def test_the_repos_metadata_without_the_archive_is_not_installed(
        self, tmp_path: Path,
    ) -> None:
        _write_transformers_snapshot(
            tmp_path, {"packed_manifest.json": "{}", "fermion_container.py": "", "NOTICE": ""},
        )

        assert _example_model_is_installed(tmp_path) is False

    def test_an_archive_still_downloading_is_not_installed(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "models--example--transformers" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").touch()
        (snapshot / "phonon-2.bps.tar.zst").write_text("first chunks")

        assert _example_model_is_installed(tmp_path) is False



class TestBarkInstallation:
    """Underdog Bark's small models ship one model.bark and no config.json."""

    def test_a_snapshot_holding_model_bark_is_installed(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models--example--transformers"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "model.bark").write_bytes(b"bark")
        (snapshot / "NOTICE").write_text("CC-BY-4.0")
        _write_refs_main(model_dir)

        assert _example_model_is_installed(tmp_path) is True

    def test_the_notice_alone_is_not_installed(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models--example--transformers"
        snapshot = model_dir / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        (snapshot / "NOTICE").write_text("CC-BY-4.0")
        _write_refs_main(model_dir)

        assert _example_model_is_installed(tmp_path) is False

class TestShardedSafetensorsInstallation:
    """``model.safetensors.index.json`` is a manifest, so its presence
    proves nothing about the weights: a cancelled 8 GB download leaves
    the index on disk beside zero shards. Only every shard the index
    names makes the snapshot loadable.
    """

    def test_should_accept_snapshot_holding_every_shard(
        self, tmp_path: Path,
    ) -> None:
        _write_sharded_snapshot(tmp_path, shards_on_disk=_SHARD_NAMES)

        assert _example_model_is_installed(tmp_path) is True

    def test_should_reject_index_with_no_shard_on_disk(
        self, tmp_path: Path,
    ) -> None:
        _write_sharded_snapshot(tmp_path, shards_on_disk=())

        assert _example_model_is_installed(tmp_path) is False

    def test_should_reject_index_with_only_some_shards_on_disk(
        self, tmp_path: Path,
    ) -> None:
        _write_sharded_snapshot(tmp_path, shards_on_disk=_SHARD_NAMES[:1])

        assert _example_model_is_installed(tmp_path) is False

    def test_should_reject_unparseable_index(self, tmp_path: Path) -> None:
        _write_sharded_snapshot(
            tmp_path, shards_on_disk=_SHARD_NAMES, index_text="{ not json",
        )

        assert _example_model_is_installed(tmp_path) is False

    def test_should_reject_index_without_a_weight_map(
        self, tmp_path: Path,
    ) -> None:
        _write_sharded_snapshot(
            tmp_path,
            shards_on_disk=_SHARD_NAMES,
            index_text=json.dumps({"metadata": {"total_size": 8}}),
        )

        assert _example_model_is_installed(tmp_path) is False


class TestPinnedModelInstallation:
    """A pinned model counts as installed only when ``refs/main`` names
    the pinned commit. ``refs/main`` is written last, after every file
    lands, so it is the completion barrier - reading the snapshot
    directory instead would call a half-finished download installed.
    """

    def test_should_accept_snapshot_when_refs_main_names_the_pin(
        self, tmp_path: Path,
    ) -> None:
        _write_pinned_snapshot(tmp_path)

        assert _pinned_manager(tmp_path).is_installed(_PINNED_MODEL.name) is True

    def test_should_reject_legacy_cache_at_a_different_commit(
        self, tmp_path: Path,
    ) -> None:
        _write_pinned_snapshot(
            tmp_path, snapshot_sha=_LEGACY_SHA, refs_sha=_LEGACY_SHA,
        )

        assert _pinned_manager(tmp_path).is_installed(_PINNED_MODEL.name) is False

    def test_should_reject_complete_pinned_snapshot_without_refs_main(
        self, tmp_path: Path,
    ) -> None:
        _write_pinned_snapshot(tmp_path, refs_sha="")

        assert _pinned_manager(tmp_path).is_installed(_PINNED_MODEL.name) is False

    def test_should_reject_pinned_snapshot_while_refs_main_still_lags(
        self, tmp_path: Path,
    ) -> None:
        # Re-download over a legacy cache: the pinned snapshot directory
        # is already on disk while ``refs/main`` still names the old
        # commit, so the pinned files may be mid-flight.
        _write_pinned_snapshot(tmp_path, refs_sha=_LEGACY_SHA)

        assert _pinned_manager(tmp_path).is_installed(_PINNED_MODEL.name) is False


class TestHuggingFaceModelManagerOperations:
    def test_get_model_path_should_reflect_installation(
        self, tmp_path: Path,
    ) -> None:
        assert _make_manager(tmp_path).get_model_path(_NEMO_MODEL.name) is None
        model_dir = _write_nemo_snapshot(tmp_path)

        assert _make_manager(tmp_path).get_model_path(_NEMO_MODEL.name) == model_dir

    def test_download_should_resolve_and_enqueue(self, tmp_path: Path) -> None:
        resolver = MagicMock()
        resolver.resolve.return_value = ResolvedRepo(
            requested_revision="",
            canonical_commit="sha123",
            files=[
                ResolvedFile(
                    filename="model.nemo",
                    size=100,
                    url=(
                        "https://huggingface.co/example/"
                        "nemo/resolve/sha123/model.nemo"
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

        handle = manager.download(_NEMO_MODEL.name)

        resolver.resolve.assert_called_once_with("example/nemo", "")
        job = download_manager.enqueue.call_args.args[0]
        assert job.job_id == _NEMO_MODEL.name
        assert job.repo_id == "example/nemo"
        assert job.commit_sha == "sha123"
        assert handle is expected_handle

    def test_download_should_request_the_models_pinned_revision(
        self, tmp_path: Path,
    ) -> None:
        resolver = MagicMock()
        resolver.resolve.return_value = ResolvedRepo(
            requested_revision=_PINNED_SHA,
            canonical_commit=_PINNED_SHA,
            files=[],
        )
        download_manager = MagicMock()
        manager = _make_manager(
            tmp_path,
            resolver=resolver,
            download_manager=download_manager,
            models={_PINNED_MODEL.name: _PINNED_MODEL},
        )

        manager.download(_PINNED_MODEL.name)

        resolver.resolve.assert_called_once_with("example/pinned", _PINNED_SHA)
        assert download_manager.enqueue.call_args.args[0].commit_sha == _PINNED_SHA

    def test_delete_should_remove_model_directory(self, tmp_path: Path) -> None:
        model_dir = _write_nemo_snapshot(tmp_path)
        manager = _make_manager(tmp_path)

        manager.delete(_NEMO_MODEL.name)

        assert model_dir.exists() is False

    def test_delete_should_remove_a_pinned_model_cache(
        self, tmp_path: Path,
    ) -> None:
        model_dir = _write_pinned_snapshot(tmp_path)
        manager = _pinned_manager(tmp_path)

        manager.delete(_PINNED_MODEL.name)

        assert model_dir.exists() is False
        assert manager.is_installed(_PINNED_MODEL.name) is False

    def test_should_implement_model_manager_interface(self, tmp_path: Path) -> None:
        assert isinstance(_make_manager(tmp_path), ModelManager)


class TestLicenceGatedDownload:
    """Acquiring weights is gated: a model whose licence has not been
    accepted must not be fetched, and nothing may be enqueued or even
    resolved on its behalf.
    """

    def _gated_manager(
        self, tmp_path: Path, config: FakeConfig,
    ) -> tuple[HuggingFaceModelManager, MagicMock, MagicMock]:
        resolver = MagicMock()
        download_manager = MagicMock()
        manager = _make_manager(
            tmp_path,
            resolver=resolver,
            download_manager=download_manager,
            models=CRISPERWHISPER_MODELS,
            license_policy=LicensePolicy(config),
        )
        return manager, resolver, download_manager

    def test_should_refuse_to_download_a_model_whose_licence_is_unaccepted(
        self, tmp_path: Path,
    ) -> None:
        manager, resolver, download_manager = self._gated_manager(
            tmp_path, FakeConfig(),
        )

        with pytest.raises(LicenseNotAcceptedError, match=_GATED_MODEL_NAME):
            manager.download(_GATED_MODEL_NAME)

        assert resolver.resolve.called is False
        assert download_manager.enqueue.called is False

    def test_should_download_once_the_licence_has_been_accepted(
        self, tmp_path: Path,
    ) -> None:
        config = FakeConfig()
        manager, resolver, download_manager = self._gated_manager(
            tmp_path, config,
        )
        resolver.resolve.return_value = ResolvedRepo(
            requested_revision="", canonical_commit="sha123", files=[],
        )
        LicensePolicy(config).accept(CRISPERWHISPER_MODELS[_GATED_MODEL_NAME])

        manager.download(_GATED_MODEL_NAME)

        assert download_manager.enqueue.called is True

    def test_should_leave_an_ungated_model_downloadable(
        self, tmp_path: Path,
    ) -> None:
        resolver = MagicMock()
        resolver.resolve.return_value = ResolvedRepo(
            requested_revision="", canonical_commit="sha123", files=[],
        )
        download_manager = MagicMock()
        manager = _make_manager(
            tmp_path,
            resolver=resolver,
            download_manager=download_manager,
            license_policy=LicensePolicy(FakeConfig()),
        )

        manager.download(_NEMO_MODEL.name)

        assert download_manager.enqueue.called is True
