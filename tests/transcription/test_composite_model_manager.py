from pathlib import Path
from unittest.mock import MagicMock

from voicetotext.transcription.composite_model_manager import CompositeModelManager
from voicetotext.transcription.model_info import ModelInfo
from voicetotext.transcription.model_manager import ModelManager


def _info(name: str, repo_id: str = "openai/whisper-base") -> ModelInfo:
    return ModelInfo(
        name=name, size_mb=290, description="test", repo_id=repo_id,
    )


class _FakeManager(ModelManager):

    def __init__(
        self,
        models: list[ModelInfo] | None = None,
        installed: dict[str, bool] | None = None,
        path: Path | None = None,
    ) -> None:
        self._models = models or []
        self._installed = installed or {}
        self._path = path
        self.download_calls: list[str] = []
        self.delete_calls: list[str] = []

    def list_models(self) -> list[ModelInfo]:
        return list(self._models)

    def has_model(self, name: str) -> bool:
        return any(info.name == name for info in self._models)

    def is_installed(self, name: str) -> bool:
        return self._installed.get(name, False)

    def download(self, name: str):  # type: ignore[override]
        self.download_calls.append(name)
        return MagicMock()

    def delete(self, name: str) -> None:
        self.delete_calls.append(name)

    def get_model_path(self, name: str) -> Path | None:
        return self._path if self._installed.get(name, False) else None


def _make_composite(
    *,
    transformers: _FakeManager | None = None,
    nano_parakeet: _FakeManager | None = None,
    active: str = "transformers",
) -> CompositeModelManager:
    return CompositeModelManager(
        by_engine={
            "transformers": transformers or _FakeManager(),
            "nano_parakeet": nano_parakeet or _FakeManager(),
        },
        active_engine_provider=lambda: active,  # type: ignore[arg-type]
    )


class TestCompositeModelManagerListing:
    def test_should_aggregate_models_from_all_children_deduped(self):
        info = _info("base")
        composite = _make_composite(
            transformers=_FakeManager(models=[info]),
            nano_parakeet=_FakeManager(models=[info]),
        )

        models = composite.list_models()

        assert len(models) == 1
        assert models[0].name == "base"


class TestCompositeModelManagerInstallation:
    def test_should_report_installed_when_any_owning_child_has_model(self):
        # The loader can open the model via the engine whose files are
        # present, so install state is engine-agnostic: a model on disk
        # under the (non-active) transformers cache is still installed.
        info = _info("large-v3-turbo")
        composite = _make_composite(
            transformers=_FakeManager(
                models=[info], installed={"large-v3-turbo": True},
            ),
            nano_parakeet=_FakeManager(
                models=[info], installed={"large-v3-turbo": False},
            ),
            active="nano_parakeet",
        )

        assert composite.is_installed("large-v3-turbo") is True

    def test_should_report_not_installed_when_no_owning_child_has_files(self):
        info = _info("large-v3-turbo")
        composite = _make_composite(
            transformers=_FakeManager(
                models=[info], installed={"large-v3-turbo": False},
            ),
            nano_parakeet=_FakeManager(
                models=[info], installed={"large-v3-turbo": False},
            ),
            active="nano_parakeet",
        )

        assert composite.is_installed("large-v3-turbo") is False

    def test_should_report_installed_when_active_engine_has_model(self):
        info = _info("large-v3-turbo")
        composite = _make_composite(
            transformers=_FakeManager(
                models=[info], installed={"large-v3-turbo": False},
            ),
            nano_parakeet=_FakeManager(
                models=[info], installed={"large-v3-turbo": True},
            ),
            active="nano_parakeet",
        )

        assert composite.is_installed("large-v3-turbo") is True

    def test_should_report_not_installed_when_no_child_owns_model(self):
        composite = _make_composite()

        assert composite.is_installed("unknown") is False

    def test_should_delegate_to_owning_child_when_active_does_not_own_model(self):
        info = _info(
            "parakeet-tdt-0.6b-v3", repo_id="nvidia/parakeet-tdt-0.6b-v3",
        )
        composite = CompositeModelManager(
            by_engine={
                "transformers": _FakeManager(),
                "nano_parakeet": _FakeManager(
                    models=[info],
                    installed={"parakeet-tdt-0.6b-v3": True},
                ),
            },
            active_engine_provider=lambda: "transformers",
        )

        assert composite.is_installed("parakeet-tdt-0.6b-v3") is True


class TestCompositeModelManagerRouting:
    def test_should_route_download_to_active_engine_child(self):
        transformers = _FakeManager()
        nano_parakeet = _FakeManager()
        composite = _make_composite(
            transformers=transformers,
            nano_parakeet=nano_parakeet,
            active="nano_parakeet",
        )

        composite.download("base")

        assert nano_parakeet.download_calls == ["base"]
        assert transformers.download_calls == []

    def test_should_route_delete_to_active_engine_child(self):
        transformers = _FakeManager()
        nano_parakeet = _FakeManager()
        composite = _make_composite(
            transformers=transformers,
            nano_parakeet=nano_parakeet,
            active="transformers",
        )

        composite.delete("base")

        assert transformers.delete_calls == ["base"]
        assert nano_parakeet.delete_calls == []

    def test_should_return_get_model_path_from_active_engine_child(self, tmp_path):
        active_path = tmp_path / "models--openai--whisper-base"
        composite = _make_composite(
            transformers=_FakeManager(installed={"base": True}, path=active_path),
            active="transformers",
        )

        assert composite.get_model_path("base") == active_path

    def test_should_implement_model_manager_interface(self):
        composite = _make_composite()

        assert isinstance(composite, ModelManager)


class TestCompositeModelManagerOwnerRouting:
    def test_should_route_download_to_owning_child_when_active_does_not_own_model(self):
        info = _info("parakeet-tdt-0.6b-v2", repo_id="nvidia/parakeet-tdt-0.6b-v2")
        owner = _FakeManager(models=[info])
        non_owner = _FakeManager()
        composite = CompositeModelManager(
            by_engine={
                "transformers": non_owner,
                "nano_parakeet": owner,
            },
            active_engine_provider=lambda: "transformers",
        )

        composite.download("parakeet-tdt-0.6b-v2")

        assert owner.download_calls == ["parakeet-tdt-0.6b-v2"]
        assert non_owner.download_calls == []

    def test_should_route_delete_to_owning_child_when_active_does_not_own_model(self):
        info = _info("parakeet-tdt-0.6b-v2", repo_id="nvidia/parakeet-tdt-0.6b-v2")
        owner = _FakeManager(models=[info])
        non_owner = _FakeManager()
        composite = CompositeModelManager(
            by_engine={
                "transformers": non_owner,
                "nano_parakeet": owner,
            },
            active_engine_provider=lambda: "transformers",
        )

        composite.delete("parakeet-tdt-0.6b-v2")

        assert owner.delete_calls == ["parakeet-tdt-0.6b-v2"]
        assert non_owner.delete_calls == []

    def test_should_prefer_active_child_when_multiple_own_model(self):
        info = _info("base")
        active = _FakeManager(models=[info])
        also_owner = _FakeManager(models=[info])
        composite = CompositeModelManager(
            by_engine={
                "transformers": active,
                "nano_parakeet": also_owner,
            },
            active_engine_provider=lambda: "transformers",
        )

        composite.download("base")

        assert active.download_calls == ["base"]
        assert also_owner.download_calls == []

    def test_has_model_should_return_true_when_any_child_owns_it(self):
        info = _info("parakeet-tdt-0.6b-v2", repo_id="nvidia/parakeet-tdt-0.6b-v2")
        composite = CompositeModelManager(
            by_engine={
                "transformers": _FakeManager(),
                "nano_parakeet": _FakeManager(models=[info]),
            },
            active_engine_provider=lambda: "transformers",
        )

        assert composite.has_model("parakeet-tdt-0.6b-v2") is True
        assert composite.has_model("does-not-exist") is False
