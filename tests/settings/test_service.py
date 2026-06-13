from unittest.mock import MagicMock

import pytest

from tests.fakes.dispatcher import DeferredActionDispatcher, InlineActionDispatcher
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
    ModelDeletedEvent,
    StateChangedEvent,
)
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.download_coordinator import DownloadCoordinator
from voicetotext.settings.engine_hot_swap import EngineHotSwapCoordinator
from voicetotext.settings.model_installation_cache import ModelInstallationCache
from voicetotext.settings.service import (
    DownloadEventListener,
    EngineEventListener,
    SettingsService,
    StateChangeListener,
)
from voicetotext.settings.settings_draft import SettingsDraft
from tests.settings.fakes import FakeConfig, FakeModelManager


def _build_service(
    installed: dict[str, bool] | None = None,
    config_overrides: dict | None = None,
    state_machine: AppStateMachine | None = None,
    registry=None,
    hardware=None,
    models=None,
):
    config_data = {
        "transcription.model_size": "base",
        "transcription.language": "en",
        "transcription.device": "auto",
        "hotkey.combination": "ctrl+alt+m",
        "hotkey.mode": "toggle",
        "output.mode": "direct_paste",
    }
    config_data.update(config_overrides or {})

    config = FakeConfig(config_data)
    manager = FakeModelManager(installed=installed or {}, models=models)
    bus = SimpleEventBus()
    cache = ModelInstallationCache(manager, bus)
    downloads = DownloadCoordinator(manager)
    hot_swap = EngineHotSwapCoordinator(
        use_case=MagicMock(),
        event_bus=bus,
        loader=lambda: MagicMock(),
        action_dispatcher=InlineActionDispatcher(),
    )
    if state_machine is None:
        state_machine = AppStateMachine(bus)
    service = SettingsService(
        config=config,
        model_manager=manager,
        event_bus=bus,
        install_cache=cache,
        downloads=downloads,
        hot_swap=hot_swap,
        state_machine=state_machine,
        registry=registry,
        hardware=hardware,
    )
    return service, config, manager, bus, hot_swap


class TestEngineCandidatesFor:
    def test_should_return_empty_tuple_when_registry_not_wired(self):
        service, *_ = _build_service()

        assert service.engine_candidates_for("base") == ()

    def test_should_return_ranked_candidates_when_registry_wired(self):
        from voicetotext.transcription.engine_factory import build_engine_registry
        from voicetotext.transcription.engine_types import HardwareProfile

        registry = build_engine_registry()
        hardware = HardwareProfile(
            kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200",
        )
        service, *_ = _build_service(registry=registry, hardware=hardware)

        candidates = service.engine_candidates_for("base")

        assert len(candidates) == 2
        assert candidates[0].is_recommended is True
        assert candidates[0].engine_id == "faster_whisper"

    def test_should_return_empty_when_model_unknown(self):
        from voicetotext.transcription.engine_factory import build_engine_registry
        from voicetotext.transcription.engine_types import HardwareProfile

        registry = build_engine_registry()
        hardware = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)
        service, *_ = _build_service(registry=registry, hardware=hardware)

        assert service.engine_candidates_for("nonexistent-model") == ()


class TestAvailableDevices:
    def test_should_return_cpu_only_when_hardware_not_wired(self):
        service, *_ = _build_service()

        assert service.available_devices() == ["auto", "cpu"]

    def test_should_omit_cuda_for_cpu_only_hardware(self):
        from voicetotext.transcription.engine_types import HardwareProfile

        cpu_only = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)
        service, *_ = _build_service(hardware=cpu_only)

        assert service.available_devices() == ["auto", "cpu"]

    def test_should_include_cuda_for_amd_rocm_hardware(self):
        from voicetotext.transcription.engine_types import HardwareProfile

        rocm = HardwareProfile(
            kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200",
        )
        service, *_ = _build_service(hardware=rocm)

        assert service.available_devices() == ["auto", "cuda", "cpu"]

    def test_should_include_cuda_for_nvidia_hardware(self):
        from voicetotext.transcription.engine_types import HardwareProfile

        cuda = HardwareProfile(
            kind="nvidia_cuda", device_name="NVIDIA RTX 4090", gfx_arch=None,
        )
        service, *_ = _build_service(hardware=cuda)

        assert service.available_devices() == ["auto", "cuda", "cpu"]


class TestAllowsDeviceSelection:
    def test_should_allow_device_selection_for_base(self):
        service, *_ = _build_service()

        assert service.allows_device_selection("base") is True

    def test_should_disallow_device_selection_for_large_v3(self):
        service, *_ = _build_service()

        assert service.allows_device_selection("large-v3") is False


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


class TestIsInstalled:
    def test_should_return_true_when_cache_says_installed(self):
        service, *_ = _build_service(installed={"base": True})

        assert service.is_installed("base") is True

    def test_should_return_false_when_cache_says_not_installed(self):
        service, *_ = _build_service(installed={"base": False})

        assert service.is_installed("base") is False


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

    def test_should_publish_model_deleted_event_on_successful_delete(self):
        service, _, _, bus, _ = _build_service(
            config_overrides={"transcription.model_size": "base"},
            installed={"tiny": True},
        )
        received: list[ModelDeletedEvent] = []
        bus.subscribe(ModelDeletedEvent, received.append)

        service.delete_model("tiny")

        assert len(received) == 1
        assert received[0].name == "tiny"

    def test_should_not_publish_event_when_delete_rejected_for_active_model(self):
        service, _, _, bus, _ = _build_service(
            config_overrides={"transcription.model_size": "base"},
            installed={"base": True},
        )
        received: list[ModelDeletedEvent] = []
        bus.subscribe(ModelDeletedEvent, received.append)

        with pytest.raises(ValueError):
            service.delete_model("base")

        assert received == []


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
        assert draft.output_mode is OutputMode.CLIPBOARD_ONLY

    def test_should_pass_language_through_without_translation(self):
        service, *_ = _build_service(
            config_overrides={"transcription.language": "fr"}
        )

        draft = service.current_draft()

        assert draft.language == "fr"


class TestApplyDraft:
    def test_should_persist_all_fields_and_save_config(self):
        service, config, *_ = _build_service()
        draft = SettingsDraft(
            model_size="small",
            language="en",
            device="cuda",
            engine="auto",
            hotkey_combination="f12",
            hotkey_mode="push_to_talk",
            output_mode=OutputMode.CLIPBOARD_ONLY,
        )

        service.apply_draft(draft)

        assert config.get("transcription.model_size") == "small"
        assert config.get("transcription.language") == "en"
        assert config.get("transcription.device") == "cuda"
        assert config.get("hotkey.combination") == "f12"
        assert config.get("hotkey.mode") == "push_to_talk"
        assert config.get("output.mode") == "clipboard_only"
        assert config.save_count == 1

    def test_should_persist_language_exactly_as_given(self):
        service, config, *_ = _build_service()
        draft = SettingsDraft(
            model_size="base",
            language="de",
            device="auto",
            engine="auto",
            hotkey_combination="ctrl+alt+m",
            hotkey_mode="toggle",
            output_mode=OutputMode.DIRECT_PASTE,
        )

        service.apply_draft(draft)

        assert config.get("transcription.language") == "de"

    def test_should_return_changeset_reflecting_only_changed_fields(self):
        service, *_ = _build_service()

        draft = SettingsDraft(
            model_size="small",  # changed from "base"
            language="en",
            device="auto",
            engine="auto",
            hotkey_combination="ctrl+alt+m",
            hotkey_mode="toggle",
            output_mode=OutputMode.DIRECT_PASTE,
        )

        changes = service.apply_draft(draft)

        assert changes.engine_reload_required is True
        assert changes.hotkey_changed is False
        assert changes.output_changed is False

    def test_should_clear_install_cache_when_engine_changes(self):
        service, _, manager, *_ = _build_service(
            config_overrides={"transcription.engine": "transformers"},
            installed={"base": True},
        )
        # Prime the cache so a stale True would survive a non-clearing apply.
        assert service.is_installed("base") is True
        manager._installed["base"] = False

        draft = SettingsDraft(
            model_size="base",
            language="en",
            device="auto",
            engine="faster_whisper",  # changed from "transformers"
            hotkey_combination="ctrl+alt+m",
            hotkey_mode="toggle",
            output_mode=OutputMode.DIRECT_PASTE,
        )
        service.apply_draft(draft)

        assert service.is_installed("base") is False

    def test_should_preserve_install_cache_when_engine_unchanged(self):
        service, _, manager, *_ = _build_service(
            config_overrides={"transcription.engine": "transformers"},
            installed={"base": True},
        )
        assert service.is_installed("base") is True
        manager._installed["base"] = False  # simulate stale disk read

        draft = SettingsDraft(
            model_size="base",
            language="en",
            device="auto",
            engine="transformers",  # unchanged
            hotkey_combination="ctrl+alt+m",
            hotkey_mode="toggle",
            output_mode=OutputMode.DIRECT_PASTE,
        )
        service.apply_draft(draft)

        # Cache preserved → stale True still returned.
        assert service.is_installed("base") is True


class TestTriggerEngineSwap:
    def test_should_start_swap_on_coordinator(self):
        service, _, _, bus, _ = _build_service()
        started = []
        bus.subscribe(EngineLoadStartedEvent, started.append)

        service.trigger_engine_swap()

        assert len(started) == 1

    def test_should_ignore_swap_when_already_busy(self):
        service, _, _, bus, _ = _build_service()
        dispatcher = DeferredActionDispatcher()
        hot_swap = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=lambda: MagicMock(),
            action_dispatcher=dispatcher,
        )
        service._hot_swap = hot_swap

        service.trigger_engine_swap()
        service.trigger_engine_swap()  # should not raise

        dispatcher.drain_all()


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
        dispatcher = DeferredActionDispatcher()
        hot_swap = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=lambda: MagicMock(),
            action_dispatcher=dispatcher,
        )
        service = SettingsService(
            config=config,
            model_manager=manager,
            event_bus=bus,
            install_cache=cache,
            downloads=downloads,
            hot_swap=hot_swap,
            state_machine=AppStateMachine(bus),
        )

        assert service.is_engine_swapping() is False
        service.trigger_engine_swap()
        assert service.is_engine_swapping() is True

        dispatcher.drain_all()
        assert service.is_engine_swapping() is False


class TestCanApplyNow:
    def test_should_return_true_when_state_is_idle(self):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus)

        service, *_ = _build_service(state_machine=state_machine)

        assert service.can_apply_now() is True

    def test_should_return_false_when_state_is_recording(self):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus)
        state_machine.transition_to(AppState.RECORDING)

        service, *_ = _build_service(state_machine=state_machine)

        assert service.can_apply_now() is False

    def test_should_return_false_when_state_is_transcribing(self):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus)
        state_machine.transition_to(AppState.RECORDING)
        state_machine.transition_to(AppState.TRANSCRIBING)

        service, *_ = _build_service(state_machine=state_machine)

        assert service.can_apply_now() is False

    def test_should_return_true_when_state_is_error(self):
        bus = SimpleEventBus()
        state_machine = AppStateMachine(bus)
        state_machine.transition_to(AppState.ERROR)

        service, *_ = _build_service(state_machine=state_machine)

        assert service.can_apply_now() is True


class _StateListener(StateChangeListener):
    def __init__(self) -> None:
        self.events: list = []

    def on_state_changed(self, event):
        self.events.append(event)


class TestStateChangeSubscription:
    def test_should_forward_state_change_events_to_listener(self):
        service, _, _, bus, _ = _build_service()
        listener = _StateListener()

        service.subscribe_state_changes(listener)
        bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))

        assert len(listener.events) == 1
        assert listener.events[0].new_state == "recording"

    def test_should_stop_forwarding_after_unsubscribe(self):
        service, _, _, bus, _ = _build_service()
        listener = _StateListener()
        service.subscribe_state_changes(listener)
        service.unsubscribe_state_changes(listener)

        bus.publish(StateChangedEvent(old_state="idle", new_state="recording"))

        assert listener.events == []
