from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt

from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.model_view import ModelView
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

    def test_should_subscribe_engine_and_state_listeners(self, service):
        # Construction alone should subscribe.
        QtSettingsWindow(service)

        assert service.subscribe_engine.called
        assert service.subscribe_state_changes.called


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


class TestQtSettingsWindowStateMachineGuard:
    def test_should_disable_save_when_cannot_apply_now(self, qtbot, service):
        service.can_apply_now.return_value = False
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        assert window.save_button().isEnabled() is False


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
