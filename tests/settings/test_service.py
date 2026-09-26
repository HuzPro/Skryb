import json
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest

from tests.fakes.dispatcher import DeferredActionDispatcher, InlineActionDispatcher
from tests.fakes.hardware import FakeHardwareDetector
from tests.fakes.licensing import AllowAllLicensePolicy
from tests.settings.fakes import FakeConfig, FakeModelManager
from voicetotext.core.config import ConfigurationStore
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
from voicetotext.core.exceptions import LicenseNotAcceptedError
from voicetotext.core.json_config_adapter import JsonConfigAdapter
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.output.dictionary_config import (
    DictionaryConfig,
    DictionaryOptions,
    LiteralRule,
    RegexRule,
)
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
from voicetotext.transcription.license_policy import LicensePolicy
from voicetotext.transcription.model_info import ALL_MODELS

_PARAKEET = "parakeet-tdt-0.6b-v3"


def test_recording_consent_round_trips_through_saved_settings() -> None:
    service, config, _, _, _ = _build_service()
    assert service.current_draft().capture_content is False

    changes = service.apply_draft(replace(service.current_draft(), capture_content=True))

    assert config.get("diagnostics.capture_content") is True
    assert service.current_draft().capture_content is True
    assert changes.diagnostics_changed
    assert changes.any_changed
    service.apply_draft(replace(service.current_draft(), capture_content=False))
    assert config.get("diagnostics.capture_content") is False
_NEMOTRON = "nemotron-speech-streaming-en-0.6b"
_CRISPERWHISPER = {
    "crisperwhisper-2.0-turbo",
    "crisperwhisper-2.0-large",
}


def _build_service(
    installed: dict[str, bool] | None = None,
    config_overrides: dict | None = None,
    state_machine: AppStateMachine | None = None,
    registry=None,
    hardware=None,
    models=None,
    config_store: ConfigurationStore | None = None,
    dictionary_path: Path | None = None,
    license_policy=None,
):
    config_data = {
        "transcription.model_size": _PARAKEET,
        "transcription.language": "en",
        "transcription.device": "auto",
        "hotkey.combination": "ctrl+alt+m",
        "hotkey.mode": "toggle",
        "output.mode": "direct_paste",
    }
    config_data.update(config_overrides or {})

    config = config_store if config_store is not None else FakeConfig(config_data)
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
        dictionary_path=dictionary_path or Path("dictionary.json"),
        license_policy=license_policy or AllowAllLicensePolicy(),
        registry=registry,
        hardware_detector=(
            None if hardware is None else FakeHardwareDetector(hardware)
        ),
    )
    return service, config, manager, bus, hot_swap


class TestEngineCandidatesFor:
    def test_should_return_empty_tuple_when_registry_not_wired(self):
        service, *_ = _build_service()

        assert service.engine_candidates_for(_PARAKEET) == ()

    def test_should_return_ranked_candidates_when_registry_wired(self):
        from voicetotext.transcription.engine_factory import build_engine_registry
        from voicetotext.transcription.engine_types import HardwareProfile

        registry = build_engine_registry()
        hardware = HardwareProfile(
            kind="amd_rocm", device_name="AMD Radeon RX 9060 XT", gfx_arch="gfx1200",
        )
        service, *_ = _build_service(registry=registry, hardware=hardware)

        candidates = service.engine_candidates_for(_PARAKEET)

        assert len(candidates) == 1
        assert candidates[0].is_recommended is True
        assert candidates[0].engine_id == "nano_parakeet"

    def test_should_return_empty_when_model_unknown(self):
        from voicetotext.transcription.engine_factory import build_engine_registry
        from voicetotext.transcription.engine_types import HardwareProfile

        registry = build_engine_registry()
        hardware = HardwareProfile(kind="cpu", device_name="CPU", gfx_arch=None)
        service, *_ = _build_service(registry=registry, hardware=hardware)

        assert service.engine_candidates_for("nonexistent-model") == ()


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
        assert names == {
            _PARAKEET, _NEMOTRON, *_CRISPERWHISPER,
        }

    def test_should_mark_installed_status_on_view(self):
        service, *_ = _build_service(
            installed={_PARAKEET: True, _NEMOTRON: False},
        )

        views = {v.name: v for v in service.list_models()}

        assert views[_PARAKEET].installed is True
        assert views[_NEMOTRON].installed is False


class TestAnyModelInstalled:
    def test_should_return_false_when_no_model_installed(self):
        service, *_ = _build_service(installed={})

        assert service.any_model_installed() is False

    def test_should_return_true_when_at_least_one_installed(self):
        service, *_ = _build_service(installed={_PARAKEET: True})

        assert service.any_model_installed() is True


class TestIsInstalled:
    def test_should_return_true_when_cache_says_installed(self):
        service, *_ = _build_service(installed={_PARAKEET: True})

        assert service.is_installed(_PARAKEET) is True

    def test_should_return_false_when_cache_says_not_installed(self):
        service, *_ = _build_service(installed={_PARAKEET: False})

        assert service.is_installed(_PARAKEET) is False


class TestSelectedModel:
    def test_should_read_from_config(self):
        service, *_ = _build_service(
            config_overrides={"transcription.model_size": _NEMOTRON}
        )

        assert service.selected_model() == _NEMOTRON


class TestStartDownload:
    def test_should_delegate_to_coordinator(self):
        service, _, manager, *_ = _build_service()

        service.start_download("test-model")

        assert manager.download_calls == ["test-model"]


class TestDeleteModel:
    def test_should_delete_via_manager_when_not_selected(self):
        service, _, manager, *_ = _build_service(
            config_overrides={"transcription.model_size": _PARAKEET},
            installed={_NEMOTRON: True},
        )

        service.delete_model("test-model")

        assert manager.delete_calls == ["test-model"]

    def test_should_raise_value_error_when_deleting_active_model(self):
        service, _, manager, *_ = _build_service(
            config_overrides={"transcription.model_size": "parakeet-tdt-0.6b-v3"},
            installed={"parakeet-tdt-0.6b-v3": True},
        )

        with pytest.raises(ValueError, match="active"):
            service.delete_model("parakeet-tdt-0.6b-v3")

        assert manager.delete_calls == []

    def test_should_invalidate_cache_entry_after_delete(self):
        service, _, _manager, *_ = _build_service(
            config_overrides={"transcription.model_size": "parakeet-tdt-0.6b-v3"},
            installed={"test-model": True},
        )
        # Prime cache
        assert service.list_models()

        service.delete_model(_NEMOTRON)

        views = {v.name: v for v in service.list_models()}
        assert views[_NEMOTRON].installed is False

    def test_should_publish_model_deleted_event_on_successful_delete(self):
        service, _, _, bus, _ = _build_service(
            config_overrides={"transcription.model_size": "parakeet-tdt-0.6b-v3"},
            installed={"test-model": True},
        )
        received: list[ModelDeletedEvent] = []
        bus.subscribe(ModelDeletedEvent, received.append)

        service.delete_model("test-model")

        assert len(received) == 1
        assert received[0].name == "test-model"

    def test_should_not_publish_event_when_delete_rejected_for_active_model(self):
        service, _, _, bus, _ = _build_service(
            config_overrides={"transcription.model_size": "parakeet-tdt-0.6b-v3"},
            installed={"parakeet-tdt-0.6b-v3": True},
        )
        received: list[ModelDeletedEvent] = []
        bus.subscribe(ModelDeletedEvent, received.append)

        with pytest.raises(ValueError):
            service.delete_model("parakeet-tdt-0.6b-v3")

        assert received == []


class TestCurrentDraft:
    def test_should_project_config_into_draft(self):
        service, *_ = _build_service(
            config_overrides={
                "transcription.model_size": "nemotron-speech-streaming-en-0.6b",
                "transcription.language": "en",
                "transcription.device": "cuda",
                "hotkey.combination": "f12",
                "hotkey.mode": "push_to_talk",
                "output.mode": "clipboard_only",
            }
        )

        draft = service.current_draft()

        assert draft.model_size == "nemotron-speech-streaming-en-0.6b"
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
            model_size="nemotron-speech-streaming-en-0.6b",
            language="en",
            device="cuda",
            engine="auto",
            hotkey_combination="f12",
            hotkey_mode="push_to_talk",
            output_mode=OutputMode.CLIPBOARD_ONLY,
        )

        service.apply_draft(draft)

        assert config.get("transcription.model_size") == "nemotron-speech-streaming-en-0.6b"
        assert config.get("transcription.language") == "en"
        assert config.get("transcription.device") == "cuda"
        assert config.get("hotkey.combination") == "f12"
        assert config.get("hotkey.mode") == "push_to_talk"
        assert config.get("output.mode") == "clipboard_only"
        assert config.save_count == 1

    def test_should_persist_language_exactly_as_given(self):
        service, config, *_ = _build_service()
        draft = SettingsDraft(
            model_size="parakeet-tdt-0.6b-v3",
            language="de",
            device="auto",
            engine="auto",
            hotkey_combination="ctrl+alt+m",
            hotkey_mode="toggle",
            output_mode=OutputMode.DIRECT_PASTE,
        )

        service.apply_draft(draft)

        assert config.get("transcription.language") == "de"

    def test_should_report_a_model_change_as_reload_and_output_rebuild(self):
        service, *_ = _build_service()

        draft = SettingsDraft(
            model_size="nemotron-speech-streaming-en-0.6b",  # changed from "parakeet-tdt-0.6b-v3"
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
        assert changes.output_changed is True

    def test_should_clear_install_cache_when_engine_changes(self):
        service, _, manager, *_ = _build_service(
            config_overrides={"transcription.engine": "nemotron_streaming"},
            installed={"parakeet-tdt-0.6b-v3": True},
        )
        # Prime the cache so a stale True would survive a non-clearing apply.
        assert service.is_installed("parakeet-tdt-0.6b-v3") is True
        manager._installed["parakeet-tdt-0.6b-v3"] = False

        draft = SettingsDraft(
            model_size="parakeet-tdt-0.6b-v3",
            language="en",
            device="auto",
            engine="nano_parakeet",
            hotkey_combination="ctrl+alt+m",
            hotkey_mode="toggle",
            output_mode=OutputMode.DIRECT_PASTE,
        )
        service.apply_draft(draft)

        assert service.is_installed("parakeet-tdt-0.6b-v3") is False

    def test_should_preserve_install_cache_when_engine_unchanged(self):
        service, _, manager, *_ = _build_service(
            config_overrides={"transcription.engine": "nano_parakeet"},
            installed={"parakeet-tdt-0.6b-v3": True},
        )
        assert service.is_installed("parakeet-tdt-0.6b-v3") is True
        manager._installed["parakeet-tdt-0.6b-v3"] = False  # simulate stale disk read

        draft = SettingsDraft(
            model_size="parakeet-tdt-0.6b-v3",
            language="en",
            device="auto",
            engine="nano_parakeet",
            hotkey_combination="ctrl+alt+m",
            hotkey_mode="toggle",
            output_mode=OutputMode.DIRECT_PASTE,
        )
        service.apply_draft(draft)

        # Cache preserved → stale True still returned.
        assert service.is_installed("parakeet-tdt-0.6b-v3") is True


class TestSettingsTransfer:
    def test_should_export_current_config_as_json(self, tmp_path: Path) -> None:
        service, *_ = _build_service()
        path = tmp_path / "settings.json"

        service.export_settings(path)

        exported = json.loads(path.read_text(encoding="utf-8"))
        assert exported["transcription"]["model_size"] == _PARAKEET
        assert exported["hotkey"]["mode"] == "toggle"

    def test_should_import_unknown_keys_and_fill_missing_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        service, config, *_ = _build_service()
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({
                "transcription": {"language": "fr"},
                "future": {"nested": {"option": "enabled"}},
            }),
            encoding="utf-8",
        )

        service.import_settings(path)

        assert config.get("transcription.language") == "fr"
        assert config.get("transcription.device") == "auto"
        assert config.get("hotkey.mode") == "toggle"
        assert config.get("future.nested.option") == "enabled"
        assert config.save_count == 1

    def test_should_record_import_in_config_history(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config = JsonConfigAdapter(config_path)
        config.load()
        config.save()
        service, *_ = _build_service(config_store=config)
        import_path = tmp_path / "settings.json"
        import_path.write_text(
            json.dumps({"hotkey": {"mode": "push_to_talk"}}),
            encoding="utf-8",
        )

        service.import_settings(import_path)

        history_lines = config_path.with_name("config-history.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        entry = cast(dict[str, object], json.loads(history_lines[-1]))
        assert entry["changes"] == [
            {"key": "hotkey.mode", "from": "toggle", "to": "push_to_talk"}
        ]

    @pytest.mark.parametrize("value", [[], "settings", 1, None])
    def test_should_reject_import_when_json_is_not_an_object(
        self,
        tmp_path: Path,
        value: object,
    ) -> None:
        service, config, *_ = _build_service()
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(value), encoding="utf-8")

        with pytest.raises(TypeError, match="JSON object"):
            service.import_settings(path)

        assert config.save_count == 0


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


class TestLicenceConsent:
    """The facade exposes the three operations the UI needs - ask, show,
    record - and refuses a hot-swap onto a model whose terms are not
    accepted, so persisting a selection cannot smuggle a gated model
    into the loader.
    """

    _GATED = "crisperwhisper-2.0-turbo"

    def _gated_service(self, config: FakeConfig):
        service, *_ = _build_service(
            config_overrides={"transcription.model_size": self._GATED},
            installed={self._GATED: True},
            license_policy=LicensePolicy(config),
        )
        return service

    def test_should_report_acceptance_required_for_a_gated_model(self):
        service = self._gated_service(FakeConfig())

        assert service.license_acceptance_required(self._GATED) is True

    def test_should_report_no_acceptance_required_for_an_ungated_model(self):
        service = self._gated_service(FakeConfig())

        assert service.license_acceptance_required(_PARAKEET) is False

    def test_should_report_no_acceptance_required_for_an_unknown_model(self):
        service = self._gated_service(FakeConfig())

        assert service.license_acceptance_required("not-in-the-catalog") is False

    def test_should_expose_the_shipped_terms_for_a_gated_model(self):
        service = self._gated_service(FakeConfig())

        terms = service.license_terms(self._GATED)

        assert terms.license_id == "crisperwhisper-nc"
        assert terms.url == ALL_MODELS[self._GATED].license_url
        assert terms.text.startswith("# CrisperWhisper 2.0")

    def test_should_clear_the_requirement_once_the_licence_is_accepted(self):
        service = self._gated_service(FakeConfig())

        service.accept_license(self._GATED)

        assert service.license_acceptance_required(self._GATED) is False

    def test_should_refuse_a_hot_swap_onto_an_unaccepted_gated_model(self):
        service = self._gated_service(FakeConfig())

        with pytest.raises(LicenseNotAcceptedError, match=self._GATED):
            service.trigger_engine_swap()

    def test_should_allow_a_hot_swap_once_the_licence_is_accepted(self):
        config = FakeConfig()
        service = self._gated_service(config)
        service.accept_license(self._GATED)

        service.trigger_engine_swap()

        assert service.is_engine_swapping() is False

    def test_should_leave_a_hot_swap_onto_an_ungated_model_alone(self):
        service, _, _, bus, _ = _build_service(
            license_policy=LicensePolicy(FakeConfig()),
        )
        started = []
        bus.subscribe(EngineLoadStartedEvent, started.append)

        service.trigger_engine_swap()

        assert len(started) == 1


class TestListenerFanout:
    def test_should_forward_download_progress_to_listeners(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)

        bus.publish(
            DownloadProgressEvent(
                job_id="parakeet-tdt-0.6b-v3",
                bytes_done=100,
                bytes_total=1000,
                speed_bps=50.0,
                eta_seconds=18.0,
            )
        )

        assert len(listener.progress) == 1
        assert listener.progress[0].job_id == "parakeet-tdt-0.6b-v3"

    def test_should_forward_download_state_changes(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)

        bus.publish(DownloadStateChangedEvent(job_id="parakeet-tdt-0.6b-v3", state="paused"))

        assert len(listener.state) == 1

    def test_should_forward_download_completed(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)

        bus.publish(DownloadCompletedEvent(job_id="parakeet-tdt-0.6b-v3"))

        assert len(listener.completed) == 1

    def test_should_forward_download_failed(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)

        bus.publish(DownloadFailedEvent(job_id="parakeet-tdt-0.6b-v3", error="disk full"))

        assert len(listener.failed) == 1
        assert listener.failed[0].error == "disk full"

    def test_should_forward_engine_load_events(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_engine(listener)

        bus.publish(EngineLoadStartedEvent(model_name="nemotron-speech-streaming-en-0.6b"))
        bus.publish(EngineLoadCompletedEvent(model_name="nemotron-speech-streaming-en-0.6b"))
        bus.publish(EngineLoadFailedEvent(model_name="nemotron-speech-streaming-en-0.6b", error="oom"))

        assert len(listener.load_started) == 1
        assert len(listener.load_completed) == 1
        assert len(listener.load_failed) == 1

    def test_should_stop_forwarding_after_unsubscribe(self):
        service, _, _, bus, _ = _build_service()
        listener = _RecordingListener()
        service.subscribe_downloads(listener)
        service.unsubscribe_downloads(listener)

        bus.publish(DownloadCompletedEvent(job_id="parakeet-tdt-0.6b-v3"))

        assert len(listener.completed) == 0


class TestIsEngineSwapping:
    def test_should_reflect_hot_swap_coordinator_busy_state(self):
        config = FakeConfig({
            "transcription.model_size": "parakeet-tdt-0.6b-v3",
            "transcription.device": "auto",
            "hotkey.combination": "ctrl+alt+m",
            "hotkey.mode": "toggle",
            "output.mode": "direct_paste",
        })
        manager = FakeModelManager(installed={"parakeet-tdt-0.6b-v3": True})
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
            dictionary_path=Path("dictionary.json"),
            license_policy=AllowAllLicensePolicy(),
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


class TestDictionaryRules:
    def test_should_return_empty_rules_when_file_is_absent(
        self, tmp_path: Path,
    ) -> None:
        service, *_ = _build_service(
            dictionary_path=tmp_path / "dictionary.json",
        )

        assert service.dictionary_rules() == DictionaryConfig()

    def test_should_round_trip_saved_v2_dictionary(self, tmp_path: Path) -> None:
        path = tmp_path / "dictionary.json"
        service, *_ = _build_service(dictionary_path=path)
        dictionary = DictionaryConfig(
            options=DictionaryOptions(remove_fillers=True),
            rules=(
                LiteralRule(("god", "oh god"), "God", case_sensitive=True),
                RegexRule(r"\bum+\b", ""),
            ),
        )

        service.save_dictionary_rules(dictionary)

        assert service.dictionary_rules() == dictionary

    def test_should_write_indented_v2_json(self, tmp_path: Path) -> None:
        path = tmp_path / "dictionary.json"
        service, *_ = _build_service(dictionary_path=path)

        service.save_dictionary_rules(DictionaryConfig())

        assert path.read_text(encoding="utf-8").startswith('{\n  "version": 2')

    def test_should_create_missing_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "dictionary.json"
        service, *_ = _build_service(dictionary_path=path)

        service.save_dictionary_rules(DictionaryConfig())

        assert path.exists()


def test_microphone_and_boost_choices_round_trip_through_saved_settings() -> None:
    service, config, _, _, _ = _build_service()
    draft = service.current_draft()
    assert draft.input_device == ""
    assert draft.boost_quiet_speech is True

    service.apply_draft(
        replace(draft, input_device="Microphone (Maono PD300X Microp", boost_quiet_speech=False),
    )

    assert config.get("audio.input_device") == "Microphone (Maono PD300X Microp"
    assert config.get("audio.boost_quiet_speech") is False
    assert service.current_draft().input_device == "Microphone (Maono PD300X Microp"
    assert service.current_draft().boost_quiet_speech is False


def test_history_retention_and_paste_methods_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    store = JsonConfigAdapter(path)
    store.load()
    service, *_ = _build_service(config_store=store)
    assert service.current_draft().history_retention_days == 0
    assert service.current_draft().paste_method_overrides == ()

    changes = service.apply_draft(replace(
        service.current_draft(),
        history_retention_days=30,
        paste_method_overrides=(("mintty.exe", "ctrl_shift_v"), ("vmconnect.exe", "type")),
    ))
    reloaded = JsonConfigAdapter(path)
    reloaded.load()
    restarted, *_ = _build_service(config_store=reloaded)

    assert changes.history_retention_changed
    assert restarted.current_draft().history_retention_days == 30
    assert restarted.current_draft().paste_method_overrides == (
        ("mintty.exe", "ctrl_shift_v"), ("vmconnect.exe", "type"),
    )
    assert json.loads(path.read_text(encoding="utf-8"))["output"][
        "paste_method_overrides"
    ] == {"mintty.exe": "ctrl_shift_v", "vmconnect.exe": "type"}


def test_removing_every_paste_override_clears_them_all(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"output": {"paste_method_overrides": {
        "Mintty.exe": "ctrl_v", "broken.exe": "middle_click",
    }}}), encoding="utf-8")
    store = JsonConfigAdapter(path)
    store.load()
    service, *_ = _build_service(config_store=store)
    assert service.current_draft().paste_method_overrides == (("mintty.exe", "ctrl_v"),)

    service.apply_draft(replace(service.current_draft(), paste_method_overrides=()))

    assert store.get("output.paste_method_overrides") == {}
    assert service.current_draft().history_retention_days == 0
