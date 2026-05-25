import threading
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt

from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
    ModelDeletedEvent,
)
from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.qt_engine_bridge import QtEngineBridge
from voicetotext.settings.qt_state_bridge import QtStateBridge
from voicetotext.settings.qt_window import QtSettingsWindow
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft


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
    svc.apply_draft.return_value = ChangeSet(
        engine_reload_required=False,
        hotkey_changed=False,
        output_changed=False,
    )
    return svc


@pytest.fixture
def window(qtbot, service):
    w = QtSettingsWindow(service)
    qtbot.addWidget(w)
    return w


class TestQtSettingsWindowInitial:
    def test_should_populate_pages_from_current_draft(self, window):
        assert window.transcription_page().current_model() == "base"
        assert window.hotkeys_page().current_combination() == "ctrl+alt+m"
        assert window.output_page().current_mode() is OutputMode.DIRECT_PASTE

    def test_should_expose_save_button(self, window):
        assert window.save_button() is not None

    def test_should_subscribe_bridges_for_engine_and_state(self, qtbot, service):
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        (engine_listener,), _ = service.subscribe_engine.call_args
        (state_listener,), _ = service.subscribe_state_changes.call_args
        assert isinstance(engine_listener, QtEngineBridge)
        assert isinstance(state_listener, QtStateBridge)


class TestQtSettingsWindowCollectDraft:
    def test_should_collect_draft_from_all_pages(self, window):
        window.hotkeys_page().set_combination("ctrl+shift+space")
        window.hotkeys_page().set_mode("push_to_talk")
        window.output_page().set_mode(OutputMode.CLIPBOARD_ONLY)

        draft = window.collect_draft()

        assert draft.hotkey_combination == "ctrl+shift+space"
        assert draft.hotkey_mode == "push_to_talk"
        assert draft.output_mode is OutputMode.CLIPBOARD_ONLY


class TestQtSettingsWindowSave:
    def test_should_call_apply_draft_when_save_clicked(self, qtbot, window, service):
        qtbot.mouseClick(window.save_button(), Qt.LeftButton)

        assert service.apply_draft.called

    def test_should_invoke_on_save_callback_with_change_set(
        self, qtbot, service
    ):
        received: list[ChangeSet] = []
        changes = ChangeSet(
            engine_reload_required=False,
            hotkey_changed=True,
            output_changed=False,
        )
        service.apply_draft.return_value = changes
        window = QtSettingsWindow(service, on_save=received.append)
        qtbot.addWidget(window)

        qtbot.mouseClick(window.save_button(), Qt.LeftButton)

        assert received == [changes]

    def test_should_trigger_engine_swap_when_reload_required(
        self, qtbot, service
    ):
        service.apply_draft.return_value = ChangeSet(
            engine_reload_required=True,
            hotkey_changed=False,
            output_changed=False,
        )
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        qtbot.mouseClick(window.save_button(), Qt.LeftButton)

        assert service.trigger_engine_swap.called

    def test_should_disable_save_button_while_engine_loading(
        self, qtbot, service
    ):
        service.apply_draft.return_value = ChangeSet(
            engine_reload_required=True,
            hotkey_changed=False,
            output_changed=False,
        )
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        qtbot.mouseClick(window.save_button(), Qt.LeftButton)

        assert window.save_button().isEnabled() is False

    def test_should_re_enable_save_button_on_engine_load_completed(
        self, qtbot, service
    ):
        received: list[ChangeSet] = []
        changes = ChangeSet(
            engine_reload_required=True,
            hotkey_changed=False,
            output_changed=False,
        )
        service.apply_draft.return_value = changes
        window = QtSettingsWindow(service, on_save=received.append)
        qtbot.addWidget(window)

        qtbot.mouseClick(window.save_button(), Qt.LeftButton)
        window.on_engine_load_completed(
            EngineLoadCompletedEvent(model_name="base")
        )

        assert window.save_button().isEnabled() is True
        assert received == [changes]

    def test_should_show_error_on_engine_load_failed(self, qtbot, service):
        service.apply_draft.return_value = ChangeSet(
            engine_reload_required=True,
            hotkey_changed=False,
            output_changed=False,
        )
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        qtbot.mouseClick(window.save_button(), Qt.LeftButton)
        window.on_engine_load_failed(
            EngineLoadFailedEvent(model_name="base", error="boom")
        )

        assert window.save_button().isEnabled() is True
        assert "boom" in window.status_label().text()

    def test_should_marshal_engine_completed_from_worker_thread_to_gui(
        self, qtbot, service
    ):
        # Regression: before the Qt-bridge wiring, the hot-swap worker
        # thread called into QtSettingsWindow.on_engine_load_completed
        # directly, violating Qt thread affinity and freezing the window
        # (including the call to ``self.close()`` from _finalize_save).
        # With QtEngineBridge as a QObject child of the window, emitting
        # the signal from a worker thread resolves via Qt.AutoConnection
        # to a queued invocation on the GUI thread.
        service.apply_draft.return_value = ChangeSet(
            engine_reload_required=True,
            hotkey_changed=False,
            output_changed=False,
        )
        received: list[ChangeSet] = []
        window = QtSettingsWindow(service, on_save=received.append)
        qtbot.addWidget(window)
        qtbot.mouseClick(window.save_button(), Qt.LeftButton)
        assert window.save_button().isEnabled() is False

        event = EngineLoadCompletedEvent(model_name="base")
        with qtbot.waitSignal(
            window._engine_bridge.load_completed, timeout=1000
        ):
            threading.Thread(
                target=lambda: window._engine_bridge.on_engine_load_completed(
                    event
                ),
                daemon=True,
            ).start()

        assert received == [
            ChangeSet(
                engine_reload_required=True,
                hotkey_changed=False,
                output_changed=False,
            )
        ]


class TestQtSettingsWindowStateMachineGuard:
    def test_should_disable_save_when_cannot_apply_now(self, qtbot, service):
        service.can_apply_now.return_value = False
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        assert window.save_button().isEnabled() is False


class TestQtSettingsWindowSaveGuard:
    def test_should_block_save_when_selected_model_not_installed(
        self, qtbot, service,
    ):
        service.is_installed.return_value = False
        received: list[ChangeSet] = []
        window = QtSettingsWindow(service, on_save=received.append)
        qtbot.addWidget(window)

        qtbot.mouseClick(window.save_button(), Qt.LeftButton)

        assert received == []
        assert not service.apply_draft.called
        assert "not installed" in window.status_label().text().lower()

    def test_should_check_install_for_currently_selected_model(
        self, qtbot, service,
    ):
        service.is_installed.return_value = True
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        qtbot.mouseClick(window.save_button(), Qt.LeftButton)

        service.is_installed.assert_called_with("base")


class TestQtSettingsWindowDownloadBridge:
    def test_should_subscribe_download_bridge_to_service(self, qtbot, service):
        from voicetotext.settings.qt_download_bridge import QtDownloadBridge

        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        (download_listener,), _ = service.subscribe_downloads.call_args
        assert isinstance(download_listener, QtDownloadBridge)

    def test_should_unsubscribe_download_bridge_on_close(self, qtbot, service):
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        with qtbot.waitSignal(window.destroyed, timeout=1000):
            window.close()

        assert service.unsubscribe_downloads.called

    def test_should_propagate_download_progress_to_list_model(
        self, qtbot, service,
    ):
        from voicetotext.core.events import DownloadProgressEvent

        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        event = DownloadProgressEvent(
            job_id="small",
            bytes_done=10_000_000,
            bytes_total=100_000_000,
            speed_bps=1_000_000.0,
            eta_seconds=90.0,
        )
        window._download_bridge.on_download_progress(event)

        # The list model should have a snapshot for "small"
        list_model = window.transcription_page().list_model()
        from voicetotext.settings.qt_model_list_model import QtModelListModel
        index = list_model.index_for_name("small")
        snapshot = list_model.data(index, QtModelListModel.DOWNLOAD_ROLE)
        assert snapshot is not None
        assert snapshot.bytes_done == 10_000_000

    def test_should_clear_download_and_refresh_install_on_completed(
        self, qtbot, service,
    ):
        from voicetotext.core.events import (
            DownloadCompletedEvent,
            DownloadProgressEvent,
        )
        from voicetotext.settings.qt_model_list_model import QtModelListModel

        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        # Seed a progress snapshot first
        window._download_bridge.on_download_progress(DownloadProgressEvent(
            job_id="small",
            bytes_done=10_000_000,
            bytes_total=100_000_000,
            speed_bps=0.0,
            eta_seconds=0.0,
        ))

        window._download_bridge.on_download_completed(
            DownloadCompletedEvent(job_id="small")
        )

        list_model = window.transcription_page().list_model()
        index = list_model.index_for_name("small")
        snapshot = list_model.data(index, QtModelListModel.DOWNLOAD_ROLE)
        assert snapshot is None


class TestQtSettingsWindowModelDeleted:
    def test_should_subscribe_to_model_deleted_events_on_service(
        self, qtbot, service,
    ):
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        assert service.subscribe_model_deleted.called

    def test_should_flip_row_button_from_delete_to_download_when_model_deleted(
        self, qtbot, service,
    ):
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        # Sanity: "tiny" starts installed, button reads "Delete".
        list_model = window.transcription_page().list_model()
        index = list_model.index_for_name("tiny")
        widget = window.transcription_page().list_view().indexWidget(index)
        assert widget.action_button().text() == "Delete"

        # Service now reports "tiny" as no longer installed (simulates
        # successful delete on the model manager) and a deletion event
        # arrives on the GUI thread via the bridge.
        service.list_models.return_value = [
            _view("tiny", installed=False),
            _view("base", installed=True),
            _view("small"),
        ]
        window.on_model_deleted(ModelDeletedEvent(name="tiny"))

        widget = window.transcription_page().list_view().indexWidget(index)
        assert widget.action_button().text() == "Download"

    def test_should_unsubscribe_model_deleted_handler_on_close(
        self, qtbot, service,
    ):
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        with qtbot.waitSignal(window.destroyed, timeout=1000):
            window.close()

        assert service.unsubscribe_model_deleted.called

    def test_should_route_delete_error_signal_to_status_label(
        self, qtbot, service,
    ):
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        window.transcription_page().delete_error.emit(
            "base", "cannot delete active model 'base'",
        )

        status_text = window.status_label().text().lower()
        assert "base" in status_text
        assert "cannot delete" in status_text


class TestQtSettingsWindowFirstRun:
    def test_should_block_save_when_first_run_and_no_model(
        self, qtbot, service
    ):
        service.any_model_installed.return_value = False
        received: list[ChangeSet] = []
        window = QtSettingsWindow(
            service, first_run=True, on_save=received.append
        )
        qtbot.addWidget(window)

        qtbot.mouseClick(window.save_button(), Qt.LeftButton)

        assert received == []
        assert "model" in window.status_label().text().lower()


class TestQtSettingsWindowCaptureCallback:
    def test_should_invoke_on_capture_state_changed_when_recording_starts(
        self, qtbot, service
    ):
        events: list[bool] = []
        window = QtSettingsWindow(
            service, on_capture_state_changed=events.append
        )
        qtbot.addWidget(window)

        window.hotkeys_page().hotkey_capture().start_recording()

        assert events == [True]

    def test_should_invoke_on_capture_state_changed_when_recording_stops(
        self, qtbot, service
    ):
        events: list[bool] = []
        window = QtSettingsWindow(
            service, on_capture_state_changed=events.append
        )
        qtbot.addWidget(window)

        window.hotkeys_page().hotkey_capture().start_recording()
        window.hotkeys_page().hotkey_capture().stop_recording()

        assert events == [True, False]

    def test_should_be_safe_when_no_capture_callback_registered(
        self, qtbot, service
    ):
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        window.hotkeys_page().hotkey_capture().start_recording()
        window.hotkeys_page().hotkey_capture().stop_recording()


class TestQtSettingsWindowLifecycle:
    def test_should_destroy_window_on_close(self, qtbot, service):
        # Regression: QMainWindow by default only hides on close(), leaving
        # the window alive in memory and its bridges subscribed to the
        # service. When the holding dict later gets reassigned, Python GC
        # destroys the window without firing closeEvent — bridges stay in
        # the service listener list and raise
        # ``RuntimeError: Signal source has been deleted`` on the next
        # fanout. WA_DeleteOnClose turns close() into close-then-
        # deleteLater so the destructor runs *after* closeEvent.
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        with qtbot.waitSignal(window.destroyed, timeout=1000):
            window.close()

        assert service.unsubscribe_engine.called
        assert service.unsubscribe_state_changes.called
