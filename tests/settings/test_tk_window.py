from unittest.mock import MagicMock

import pytest

from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadProgressEvent,
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
    ModelDeletedEvent,
)
from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.signal import Signal
from voicetotext.settings.tk_download_bridge import TkDownloadBridge
from voicetotext.settings.tk_engine_bridge import TkEngineBridge
from voicetotext.settings.tk_state_bridge import TkStateBridge
from voicetotext.settings.tk_window import TkSettingsWindow
from voicetotext.settings.transcription_mode import TranscriptionMode

_PARAKEET = "parakeet-tdt-0.6b-v3"


class FakeCaptureBridge:
    def __init__(self) -> None:
        self.key_pressed = Signal()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _view(name: str, installed: bool = False) -> ModelView:
    return ModelView(
        name=name,
        display_size="74 MB",
        description=f"{name} description",
        installed=installed,
    )


def _draft(**overrides) -> SettingsDraft:
    fields = {
        "model_size": "base",
        "language": "en",
        "device": "auto",
        "engine": "auto",
        "hotkey_combination": "ctrl+alt+m",
        "hotkey_mode": "toggle",
        "output_mode": OutputMode.DIRECT_PASTE,
    }
    fields.update(overrides)
    return SettingsDraft(**fields)


def _no_change() -> ChangeSet:
    return ChangeSet(
        engine_reload_required=False, hotkey_changed=False, output_changed=False,
    )


def _reload_change() -> ChangeSet:
    return ChangeSet(
        engine_reload_required=True, hotkey_changed=False, output_changed=False,
    )


@pytest.fixture
def service():
    svc = MagicMock()
    svc.list_models.return_value = [
        _view("tiny", installed=True),
        _view("base", installed=True),
        _view("small"),
    ]
    svc.selected_model.return_value = "base"
    svc.current_draft.return_value = _draft()
    svc.any_model_installed.return_value = True
    svc.is_installed.return_value = True
    svc.can_apply_now.return_value = True
    svc.is_engine_swapping.return_value = False
    svc.available_devices.return_value = ["auto", "cuda", "cpu"]
    svc.allows_device_selection.side_effect = lambda n: n in {"tiny", "base"}
    svc.supports_streaming_hardware.return_value = True
    svc.engine_candidates_for.return_value = ()
    svc.apply_draft.return_value = _no_change()
    return svc


def _make_window(tk_root, service, **kwargs):
    return TkSettingsWindow(
        tk_root, service, capture_bridge=FakeCaptureBridge(), **kwargs
    )


@pytest.fixture
def window(tk_root, service):
    w = _make_window(tk_root, service)
    yield w
    if w.winfo_exists():
        w.destroy()


class TestInitial:
    def test_should_populate_pages_from_current_draft(self, window):
        assert window.transcription_page().current_model() == "base"
        assert window.hotkeys_page().current_combination() == "ctrl+alt+m"
        assert window.output_page().current_mode() is OutputMode.DIRECT_PASTE

    def test_should_expose_save_button(self, window):
        assert window.save_button() is not None

    def test_should_subscribe_bridges_for_engine_state_and_downloads(
        self, tk_root, service,
    ):
        _make_window(tk_root, service)

        (engine_listener,), _ = service.subscribe_engine.call_args
        (state_listener,), _ = service.subscribe_state_changes.call_args
        (download_listener,), _ = service.subscribe_downloads.call_args
        assert isinstance(engine_listener, TkEngineBridge)
        assert isinstance(state_listener, TkStateBridge)
        assert isinstance(download_listener, TkDownloadBridge)
        assert service.subscribe_model_deleted.called


class TestCollectDraft:
    def test_should_collect_draft_from_all_pages(self, window):
        window.hotkeys_page().set_combination("ctrl+shift+space")
        window.hotkeys_page().set_mode("push_to_talk")
        window.output_page().set_mode(OutputMode.CLIPBOARD_ONLY)

        draft = window.collect_draft()

        assert draft.hotkey_combination == "ctrl+shift+space"
        assert draft.hotkey_mode == "push_to_talk"
        assert draft.output_mode is OutputMode.CLIPBOARD_ONLY


class TestSave:
    def test_should_call_apply_draft_when_save_clicked(self, window, service):
        window.save_button().invoke()

        assert service.apply_draft.called

    def test_should_invoke_on_save_callback_with_change_set(self, tk_root, service):
        received: list[ChangeSet] = []
        changes = ChangeSet(
            engine_reload_required=False, hotkey_changed=True, output_changed=False,
        )
        service.apply_draft.return_value = changes
        window = _make_window(tk_root, service, on_save=received.append)

        window.save_button().invoke()

        assert received == [changes]

    def test_should_trigger_engine_swap_when_reload_required(self, tk_root, service):
        service.apply_draft.return_value = _reload_change()
        window = _make_window(tk_root, service)

        window.save_button().invoke()

        assert service.trigger_engine_swap.called

    def test_should_disable_save_button_while_engine_loading(self, tk_root, service):
        service.apply_draft.return_value = _reload_change()
        window = _make_window(tk_root, service)

        window.save_button().invoke()

        assert window.save_enabled() is False

    def test_should_re_enable_and_finalize_on_engine_load_completed(
        self, tk_root, service,
    ):
        received: list[ChangeSet] = []
        changes = _reload_change()
        service.apply_draft.return_value = changes
        window = _make_window(tk_root, service, on_save=received.append)

        window.save_button().invoke()
        window.on_engine_load_completed(EngineLoadCompletedEvent(model_name="base"))

        assert window.save_enabled() is True
        assert received == [changes]

    def test_should_show_error_on_engine_load_failed(self, tk_root, service):
        service.apply_draft.return_value = _reload_change()
        window = _make_window(tk_root, service)

        window.save_button().invoke()
        window.on_engine_load_failed(
            EngineLoadFailedEvent(model_name="base", error="boom")
        )

        assert window.save_enabled() is True
        assert "boom" in window.status_label().cget("text")

    def test_should_marshal_engine_event_through_schedule(self, tk_root, service):
        scheduled: list = []
        window = _make_window(tk_root, service, schedule=scheduled.append)

        window._engine_bridge.on_engine_load_started(
            EngineLoadStartedEvent(model_name="base")
        )

        assert window.status_label().cget("text") == ""  # deferred
        scheduled[0]()
        assert "base" in window.status_label().cget("text")


class TestStateMachineGuard:
    def test_should_disable_save_when_cannot_apply_now(self, tk_root, service):
        service.can_apply_now.return_value = False
        window = _make_window(tk_root, service)

        assert window.save_enabled() is False


class TestSaveGuard:
    def test_should_block_save_when_selected_model_not_installed(
        self, tk_root, service,
    ):
        service.is_installed.return_value = False
        received: list[ChangeSet] = []
        window = _make_window(tk_root, service, on_save=received.append)

        window.save_button().invoke()

        assert received == []
        assert not service.apply_draft.called
        assert "not installed" in window.status_label().cget("text").lower()

    def test_should_check_install_for_currently_selected_model(
        self, tk_root, service,
    ):
        window = _make_window(tk_root, service)

        window.save_button().invoke()

        service.is_installed.assert_called_with("base")


class TestDownloadBridge:
    def test_should_unsubscribe_download_bridge_on_close(self, window, service):
        window.close()

        assert service.unsubscribe_downloads.called

    def test_should_propagate_download_progress_to_model_list(self, window):
        window._download_bridge.on_download_progress(DownloadProgressEvent(
            job_id="small",
            bytes_done=10_000_000,
            bytes_total=100_000_000,
            speed_bps=1_000_000.0,
            eta_seconds=90.0,
        ))

        snapshot = window.transcription_page().model_list().download_snapshot("small")
        assert snapshot is not None
        assert snapshot.bytes_done == 10_000_000

    def test_should_clear_download_and_refresh_install_on_completed(self, window):
        window._download_bridge.on_download_progress(DownloadProgressEvent(
            job_id="small", bytes_done=10_000_000, bytes_total=100_000_000,
            speed_bps=0.0, eta_seconds=0.0,
        ))

        window._download_bridge.on_download_completed(
            DownloadCompletedEvent(job_id="small")
        )

        snapshot = window.transcription_page().model_list().download_snapshot("small")
        assert snapshot is None


class TestModelDeleted:
    def test_should_flip_row_button_to_download_when_model_deleted(
        self, window, service,
    ):
        row = window.transcription_page().model_list().row_for("tiny")
        assert row.action_button().cget("text") == "Delete"

        service.list_models.return_value = [
            _view("tiny", installed=False),
            _view("base", installed=True),
            _view("small"),
        ]
        window.on_model_deleted(ModelDeletedEvent(name="tiny"))

        assert row.action_button().cget("text") == "Download"

    def test_should_route_delete_error_to_status_label(self, window):
        window.transcription_page().delete_error.emit(
            "base", "cannot delete active model 'base'",
        )

        status = window.status_label().cget("text").lower()
        assert "base" in status
        assert "cannot delete" in status


class TestFirstRun:
    def test_should_block_save_when_first_run_and_no_model(self, tk_root, service):
        service.any_model_installed.return_value = False
        received: list[ChangeSet] = []
        window = _make_window(
            tk_root, service, first_run=True, on_save=received.append,
        )

        window.save_button().invoke()

        assert received == []
        assert "model" in window.status_label().cget("text").lower()


class TestCaptureCallback:
    def test_should_invoke_capture_callback_when_recording_starts(
        self, tk_root, service,
    ):
        events: list[bool] = []
        window = _make_window(
            tk_root, service, on_capture_state_changed=events.append,
        )

        window.hotkeys_page().hotkey_capture().start_recording()

        assert events == [True]

    def test_should_be_safe_when_no_capture_callback(self, window):
        window.hotkeys_page().hotkey_capture().start_recording()
        window.hotkeys_page().hotkey_capture().stop_recording()


class TestLifecycle:
    def test_should_unsubscribe_bridges_on_close(self, window, service):
        window.close()

        assert service.unsubscribe_engine.called
        assert service.unsubscribe_state_changes.called
        assert service.unsubscribe_model_deleted.called

    def test_should_destroy_window_after_close_when_loop_pumped(
        self, tk_root, window,
    ):
        window.close()
        tk_root.update_idletasks()

        assert not window.winfo_exists()


class TestMode:
    @pytest.fixture
    def mode_service(self):
        svc = MagicMock()

        def list_models(mode=None):
            if mode is TranscriptionMode.STREAMING:
                return [_view(_PARAKEET, installed=True)]
            return [
                _view("tiny", installed=True),
                _view("base", installed=True),
                _view(_PARAKEET, installed=True),
            ]

        svc.list_models.side_effect = list_models
        svc.selected_model.return_value = "base"
        svc.current_draft.return_value = _draft()
        svc.any_model_installed.return_value = True
        svc.is_installed.return_value = True
        svc.can_apply_now.return_value = True
        svc.is_engine_swapping.return_value = False
        svc.available_devices.return_value = ["auto", "cuda", "cpu"]
        svc.allows_device_selection.side_effect = lambda n: n in {"tiny", "base"}
        svc.model_supports_streaming.side_effect = lambda n: n == _PARAKEET
        svc.model_supports_batch.return_value = True
        svc.supports_streaming_hardware.return_value = True
        svc.engine_candidates_for.return_value = ()
        return svc

    @pytest.fixture
    def mode_window(self, tk_root, mode_service):
        w = _make_window(tk_root, mode_service)
        yield w
        if w.winfo_exists():
            w.destroy()

    def test_should_show_the_three_sidebar_sections(self, mode_window):
        assert mode_window.sidebar_labels() == ["Transcription", "Hotkeys", "Output"]

    def test_should_include_mode_in_collected_draft(self, mode_window):
        mode_window.transcription_page().apply_mode(TranscriptionMode.STREAMING)

        assert mode_window.collect_draft().transcription_mode is (
            TranscriptionMode.STREAMING
        )

    def test_should_refilter_model_list_when_user_toggles_streaming(
        self, mode_window,
    ):
        mode_window.transcription_page().mode_selector().radio_button(
            TranscriptionMode.STREAMING
        ).invoke()

        assert mode_window.transcription_page().model_list().row_names() == [
            _PARAKEET
        ]

    def test_should_announce_auto_switched_model_in_status(self, mode_window):
        mode_window.transcription_page().mode_selector().radio_button(
            TranscriptionMode.STREAMING
        ).invoke()

        status = mode_window.status_label().cget("text")
        assert _PARAKEET in status
        assert "base" in status
