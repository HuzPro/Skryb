import threading
import time
from unittest.mock import MagicMock

import pytest

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import EngineHotSwapCoordinator
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.service import (
    DownloadEventListener,
    EngineEventListener,
    SettingsService,
)
from voicetotext.settings.settings_draft import SettingsDraft
from tests.settings.fakes import FakeConfig, FakeModelManager


def _build_service(
    installed: dict[str, bool] | None = None,
    config_overrides: dict | None = None,
):
    config_data = {
        "transcription.model_size": "base",
        "transcription.language": None,
        "transcription.device": "auto",
        "hotkey.combination": "ctrl+alt+m",
        "hotkey.mode": "toggle",
        "output.mode": "direct_paste",
    }
    config_data.update(config_overrides or {})

    config = FakeConfig(config_data)
    manager = FakeModelManager(installed=installed or {})
    bus = SimpleEventBus()
    cache = ModelInstallationCache(manager, bus)
    downloads = DownloadCoordinator(manager)
    hot_swap = EngineHotSwapCoordinator(
        use_case=MagicMock(),
        event_bus=bus,
        loader=lambda: MagicMock(),
    )
    service = SettingsService(
        config=config,
        model_manager=manager,
        event_bus=bus,
        install_cache=cache,
        downloads=downloads,
        hot_swap=hot_swap,
    )
    return service, config, manager, bus, hot_swap


class _RecordingListener(DownloadEventListener, EngineEventListener):
    def __init__(self):
        self.progress: list = []
        self.state: list = []
        self.completed: list = []
        self.failed: list = []
        self.load_started: list = []
        self.load_completed: list = []
        self.load_failed: list = []

    def on_download_progress(self, event): self.progress.append(event)
    def on_download_state(self, event): self.state.append(event)
    def on_download_completed(self, event): self.completed.append(event)
    def on_download_failed(self, event): self.failed.append(event)
    def on_engine_load_started(self, event): self.load_started.append(event)
    def on_engine_load_completed(self, event): self.load_completed.append(event)
    def on_engine_load_failed(self, event): self.load_failed.append(event)


class TestListModels:
    def test_should_return_one_view_per_available_model(self):
        service, *_ = _build_service()

        views = service.list_models()

        names = {v.name for v in views}
        assert "tiny" in names
        assert "base" in names
        assert "large-v3-turbo" in names

    def test_should_mark_installed_status_on_view(self):
        service, *_ = _build_service(installed={"base": True, "tiny": False})

        views = {v.name: v for v in service.list_models()}

        assert views["base"].installed is True
        assert views["tiny"].installed is False


class TestAnyModelInstalled:
    def test_should_return_false_when_no_model_installed(self):
        service, *_ = _build_service(installed={})

        assert service.any_model_installed() is False

    def test_should_return_true_when_at_least_one_installed(self):
        service, *_ = _build_service(installed={"base": True})

        assert service.any_model_installed() is True


class TestSelectedModel:
    def test_should_read_from_config(self):
        service, *_ = _build_service(
            config_overrides={"transcription.model_size": "large-v3-turbo"}
        )

        assert service.selected_model() == "large-v3-turbo"


class TestStartDownload:
    def test_should_delegate_to_coordinator(self):
        service, _, manager, *_ = _build_service()

        service.start_download("tiny")

        assert manager.download_calls == ["tiny"]


class TestDeleteModel:
    def test_should_delete_via_manager_when_not_selected(self):
        service, _, manager, *_ = _build_service(
            config_overrides={"transcription.model_size": "base"},
            installed={"tiny": True},
        )

        service.delete_model("tiny")

        assert manager.delete_calls == ["tiny"]

    def test_should_raise_value_error_when_deleting_active_model(self):
        service, _, manager, *_ = _build_service(
            config_overrides={"transcription.model_size": "base"},
            installed={"base": True},
        )

        with pytest.raises(ValueError, match="active"):
            service.delete_model("base")

        assert manager.delete_calls == []

    def test_should_invalidate_cache_entry_after_delete(self):
        service, _, manager, *_ = _build_service(
            config_overrides={"transcription.model_size": "base"},
            installed={"tiny": True},
        )
        # Prime cache
        assert service.list_models()

        service.delete_model("tiny")

        views = {v.name: v for v in service.list_models()}
        assert views["tiny"].installed is False


class TestCurrentDraft:
    def test_should_project_config_into_draft(self):
        service, *_ = _build_service(
            config_overrides={
                "transcription.model_size": "small",
                "transcription.language": "en",
                "transcription.device": "cuda",
                "hotkey.combination": "f12",
                "hotkey.mode": "push_to_talk",
                "output.mode": "clipboard_only",
            }
        )

        draft = service.current_draft()

        assert draft.model_size == "small"
        assert draft.language == "en"
        assert draft.device == "cuda"
        assert draft.hotkey_combination == "f12"
        assert draft.hotkey_mode == "push_to_talk"
        assert draft.output_mode == "clipboard_only"

    def test_should_surface_auto_language_when_config_is_none(self):
        service, *_ = _build_service(
            config_overrides={"transcription.language": None}
        )

        draft = service.current_draft()

        assert draft.language == "auto"


class TestApplyDraft:
    def test_should_persist_all_fields_and_save_config(self):
        service, config, *_ = _build_service()
        draft = SettingsDraft(
            model_size="small",
            language="en",
            device="cuda",
            hotkey_combination="f12",
            hotkey_mode="push_to_talk",
            output_mode="clipboard_only",
        )

        service.apply_draft(draft)

        assert config.get("transcription.model_size") == "small"
        assert config.get("transcription.language") == "en"
        assert config.get("transcription.device") == "cuda"
        assert config.get("hotkey.combination") == "f12"
        assert config.get("hotkey.mode") == "push_to_talk"
        assert config.get("output.mode") == "clipboard_only"
        assert config.save_count == 1

    def test_should_store_none_for_auto_language(self):
        service, config, *_ = _build_service()
        draft = SettingsDraft(
            model_size="base",
            language="auto",
            device="auto",
            hotkey_combination="ctrl+alt+m",
            hotkey_mode="toggle",
            output_mode="direct_paste",
        )

        service.apply_draft(draft)

        assert config.get("transcription.language") is None

    def test_should_return_changeset_reflecting_only_changed_fields(self):
        service, *_ = _build_service()

        draft = SettingsDraft(
            model_size="small",  # changed from "base"
            language="auto",
            device="auto",
            hotkey_combination="ctrl+alt+m",
            hotkey_mode="toggle",
            output_mode="direct_paste",
        )

        changes = service.apply_draft(draft)

        assert changes.engine_reload_required is True
        assert changes.hotkey_changed is False
        assert changes.output_changed is False


class TestTriggerEngineSwap:
    def test_should_start_swap_on_coordinator(self):
        service, _, _, bus, hot_swap = _build_service()
        started = []
        bus.subscribe(EngineLoadStartedEvent, started.append)

        service.trigger_engine_swap()

        # The coordinator runs on a thread — give it a moment.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not started:
            time.sleep(0.01)
        assert len(started) == 1

    def test_should_ignore_swap_when_already_busy(self):
        release = threading.Event()

        def slow_loader():
            release.wait(timeout=2)
            return MagicMock()

        service, _, _, bus, _ = _build_service()
        # Replace the coordinator with one that uses a slow loader.
        hot_swap = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=slow_loader,
        )
        service._hot_swap = hot_swap

        service.trigger_engine_swap()
        time.sleep(0.05)
        service.trigger_engine_swap()  # should not raise

        release.set()


class TestListenerFanout:
    def test_should_forward_download_progress_to_listeners(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)

        bus.publish(
            DownloadProgressEvent(
                job_id="base",
                bytes_done=100,
                bytes_total=1000,
                speed_bps=50.0,
                eta_seconds=18.0,
            )
        )

        assert len(listener.progress) == 1
        assert listener.progress[0].job_id == "base"

    def test_should_forward_download_state_changes(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)

        bus.publish(DownloadStateChangedEvent(job_id="base", state="paused"))

        assert len(listener.state) == 1

    def test_should_forward_download_completed(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)

        bus.publish(DownloadCompletedEvent(job_id="base"))

        assert len(listener.completed) == 1

    def test_should_forward_download_failed(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)

        bus.publish(DownloadFailedEvent(job_id="base", error="disk full"))

        assert len(listener.failed) == 1
        assert listener.failed[0].error == "disk full"

    def test_should_forward_engine_load_events(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_engine(listener)

        bus.publish(EngineLoadStartedEvent(model_name="small"))
        bus.publish(EngineLoadCompletedEvent(model_name="small"))
        bus.publish(EngineLoadFailedEvent(model_name="small", error="oom"))

        assert len(listener.load_started) == 1
        assert len(listener.load_completed) == 1
        assert len(listener.load_failed) == 1

    def test_should_stop_forwarding_after_unsubscribe(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)
        service.unsubscribe_downloads(listener)

        bus.publish(DownloadCompletedEvent(job_id="base"))

        assert len(listener.completed) == 0


class TestIsEngineSwapping:
    def test_should_reflect_hot_swap_coordinator_busy_state(self):
        release = threading.Event()

        def slow_loader():
            release.wait(timeout=2)
            return MagicMock()

        config = FakeConfig({
            "transcription.model_size": "base",
            "transcription.device": "auto",
            "hotkey.combination": "ctrl+alt+m",
            "hotkey.mode": "toggle",
            "output.mode": "direct_paste",
        })
        manager = FakeModelManager(installed={"base": True})
        bus = SimpleEventBus()
        cache = ModelInstallationCache(manager, bus)
        downloads = DownloadCoordinator(manager)
        hot_swap = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=slow_loader,
        )
        completed = []
        bus.subscribe(EngineLoadCompletedEvent, completed.append)
        service = SettingsService(
            config=config,
            model_manager=manager,
            event_bus=bus,
            install_cache=cache,
            downloads=downloads,
            hot_swap=hot_swap,
        )

        assert service.is_engine_swapping() is False
        service.trigger_engine_swap()
        assert service.is_engine_swapping() is True

        release.set()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not completed:
            time.sleep(0.01)
        time.sleep(0.05)
        assert service.is_engine_swapping() is False
