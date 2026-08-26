from pathlib import Path
from unittest.mock import MagicMock

from voicetotext.transcription.composite_model_manager import CompositeModelManager
from voicetotext.transcription.engine_types import EngineId
from voicetotext.transcription.model_info import ModelInfo
from voicetotext.transcription.model_manager import ModelManager


def _info(name: str = "parakeet-tdt-0.6b-v3") -> ModelInfo:
    return ModelInfo(
        name=name, size_mb=2510, description="test",
        repo_id=f"example/{name}", compatible_engines=("nano_parakeet",),
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
    nano: _FakeManager | None = None,
    nemotron: _FakeManager | None = None,
    active: EngineId = "nano_parakeet",
) -> CompositeModelManager:
    return CompositeModelManager(
        by_engine={
            "nano_parakeet": nano or _FakeManager(),
            "nemotron_streaming": nemotron or _FakeManager(),
        },
        active_engine_provider=lambda: active,
    )


class TestCompositeModelManager:
    def test_should_aggregate_models_from_children_without_duplicates(
        self,
    ) -> None:
        info = _info()
        composite = _make_composite(
            nano=_FakeManager(models=[info]),
            nemotron=_FakeManager(models=[info]),
        )

        assert composite.list_models() == [info]

    def test_should_report_installed_when_any_owner_has_files(self) -> None:
        info = _info()
        composite = _make_composite(
            nano=_FakeManager(models=[info], installed={info.name: False}),
            nemotron=_FakeManager(models=[info], installed={info.name: True}),
        )

        assert composite.is_installed(info.name) is True

    def test_should_report_unknown_model_not_installed(self) -> None:
        assert _make_composite().is_installed("unknown") is False

    def test_should_route_to_non_active_owner(self) -> None:
        info = _info("nemotron-speech-streaming-en-0.6b")
        owner = _FakeManager(models=[info])
        active = _FakeManager()
        composite = _make_composite(
            nano=active, nemotron=owner, active="nano_parakeet",
        )

        composite.download(info.name)
        composite.delete(info.name)

        assert owner.download_calls == [info.name]
        assert owner.delete_calls == [info.name]
        assert active.download_calls == []

    def test_should_prefer_active_owner_when_both_own_model(self) -> None:
        info = _info()
        active = _FakeManager(models=[info])
        other = _FakeManager(models=[info])
        composite = _make_composite(nano=active, nemotron=other)

        composite.download(info.name)

        assert active.download_calls == [info.name]
        assert other.download_calls == []

    def test_get_model_path_should_use_owning_child(self, tmp_path: Path) -> None:
        info = _info()
        expected = tmp_path / "models--example--parakeet"
        composite = _make_composite(
            nano=_FakeManager(
                models=[info], installed={info.name: True}, path=expected,
            ),
        )

        assert composite.get_model_path(info.name) == expected
        assert isinstance(composite, ModelManager)
