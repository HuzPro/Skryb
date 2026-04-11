import threading

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
)
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.transcription.model_info import ModelInfo
from voicetotext.transcription.model_manager import ModelManager


class _FakeModelManager(ModelManager):
    """In-memory ModelManager stub — counts is_installed() calls."""

    def __init__(self, installed: dict[str, bool] | None = None):
        self._installed = dict(installed or {})
        self.is_installed_call_count = 0

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=name, size_mb=1, description="") for name in self._installed]

    def is_installed(self, name: str) -> bool:
        self.is_installed_call_count += 1
        return self._installed.get(name, False)

    def download(self, name):
        raise NotImplementedError

    def delete(self, name: str) -> None:
        self._installed.pop(name, None)

    def get_model_path(self, name: str):
        return None


class TestModelInstallationCacheFallback:
    def test_should_delegate_to_model_manager_on_first_call(self):
        manager = _FakeModelManager({"base": True})
        bus = SimpleEventBus()
        cache = ModelInstallationCache(manager, bus)

        assert cache.is_installed("base") is True
        assert manager.is_installed_call_count == 1

    def test_should_return_false_when_model_not_installed(self):
        manager = _FakeModelManager({"base": False})
        cache = ModelInstallationCache(manager, SimpleEventBus())

        assert cache.is_installed("base") is False


class TestModelInstallationCacheHit:
    def test_should_return_cached_value_without_hitting_model_manager(self):
        manager = _FakeModelManager({"base": True})
        cache = ModelInstallationCache(manager, SimpleEventBus())

        cache.is_installed("base")
        cache.is_installed("base")
        cache.is_installed("base")

        assert manager.is_installed_call_count == 1

    def test_should_populate_all_models_eagerly(self):
        manager = _FakeModelManager({"tiny": False, "base": True, "small": False})
        cache = ModelInstallationCache(manager, SimpleEventBus())

        cache.populate_all()

        assert cache.is_installed("tiny") is False
        assert cache.is_installed("base") is True
        assert cache.is_installed("small") is False
        # populate_all hit each once; reads after are cached.
        assert manager.is_installed_call_count == 3


class TestModelInstallationCacheInvalidation:
    def test_should_invalidate_entry_when_download_completes(self):
        manager = _FakeModelManager({"base": False})
        bus = SimpleEventBus()
        cache = ModelInstallationCache(manager, bus)

        cache.is_installed("base")  # cached as False
        manager._installed["base"] = True  # simulate on-disk change
        bus.publish(DownloadCompletedEvent(job_id="base"))

        assert cache.is_installed("base") is True

    def test_should_invalidate_entry_when_download_fails(self):
        manager = _FakeModelManager({"base": True})
        bus = SimpleEventBus()
        cache = ModelInstallationCache(manager, bus)

        cache.is_installed("base")
        manager._installed["base"] = False
        bus.publish(DownloadFailedEvent(job_id="base", error="net"))

        assert cache.is_installed("base") is False

    def test_should_invalidate_explicit_name(self):
        manager = _FakeModelManager({"base": True})
        cache = ModelInstallationCache(manager, SimpleEventBus())

        cache.is_installed("base")
        manager._installed["base"] = False
        cache.invalidate("base")

        assert cache.is_installed("base") is False


class TestModelInstallationCacheThreadSafety:
    def test_should_serialize_concurrent_populate_and_reads(self):
        manager = _FakeModelManager({"a": True, "b": False, "c": True})
        cache = ModelInstallationCache(manager, SimpleEventBus())
        errors: list[BaseException] = []

        def reader():
            try:
                for _ in range(200):
                    cache.is_installed("a")
                    cache.is_installed("b")
                    cache.is_installed("c")
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        def populator():
            try:
                for _ in range(50):
                    cache.populate_all()
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=populator))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
