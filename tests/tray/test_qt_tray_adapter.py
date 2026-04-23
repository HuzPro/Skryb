from unittest.mock import MagicMock

import pytest

from voicetotext.core.state_machine import AppState
from voicetotext.tray.qt_tray_adapter import QtTrayAdapter, build_qt_tray_adapter


@pytest.fixture
def tray_backend():
    """Stand-in for ``QSystemTrayIcon`` — we verify calls without touching the
    host tray. Avoids ``QSystemTrayIcon.isSystemTrayAvailable`` returning
    false on headless CI or Windows runners without a shell."""

    return MagicMock()


@pytest.fixture
def adapter(qtbot, tray_backend):
    get_label = MagicMock(return_value="Idle — ready")
    on_settings = MagicMock()
    on_quit = MagicMock()
    a = QtTrayAdapter(
        initial_state=AppState.IDLE,
        initial_label="Idle — ready",
        get_state_label=get_label,
        on_settings_clicked=on_settings,
        on_quit_clicked=on_quit,
        tray_backend=tray_backend,
    )
    return a


class TestQtTrayAdapterInitial:
    def test_should_set_icon_on_backend(self, adapter, tray_backend):
        assert tray_backend.setIcon.call_count == 1

    def test_should_set_tooltip_on_backend(self, adapter, tray_backend):
        tray_backend.setToolTip.assert_called_with("VoiceToText — Idle — ready")


class TestQtTrayAdapterVisibleProperty:
    def test_should_start_invisible(self, adapter):
        assert adapter.visible is False

    def test_should_show_backend_when_set_visible_true(self, adapter, tray_backend):
        adapter.visible = True

        assert adapter.visible is True
        tray_backend.show.assert_called()

    def test_should_hide_backend_when_set_visible_false(
        self, adapter, tray_backend
    ):
        adapter.visible = True
        adapter.visible = False

        tray_backend.hide.assert_called()


class TestQtTrayAdapterUpdateState:
    def test_should_set_new_icon_on_backend_when_state_updates(
        self, adapter, tray_backend
    ):
        tray_backend.reset_mock()

        adapter.update_state(AppState.RECORDING, "Recording...")

        tray_backend.setIcon.assert_called_once()

    def test_should_update_tooltip_with_title_prefix(
        self, adapter, tray_backend
    ):
        adapter.update_state(AppState.RECORDING, "Recording...")

        tray_backend.setToolTip.assert_called_with("VoiceToText — Recording...")


class TestQtTrayAdapterRunAndStop:
    def test_should_invoke_setup_callback_with_self_on_run(
        self, adapter, monkeypatch
    ):
        received: list = []
        monkeypatch.setattr(
            "voicetotext.tray.qt_tray_adapter._run_event_loop",
            lambda: None,
        )

        adapter.run(lambda icon: received.append(icon))

        assert received == [adapter]

    def test_should_call_quit_and_hide_on_stop(self, adapter, tray_backend):
        adapter.visible = True

        adapter.stop()

        tray_backend.hide.assert_called()
        assert adapter.visible is False


class TestBuildQtTrayAdapterFactory:
    def test_should_return_adapter_with_state_label_from_getter(self, qtbot):
        adapter = build_qt_tray_adapter(
            initial_state=AppState.IDLE,
            get_state_label=lambda: "Idle — ready",
            on_settings_clicked=None,
            on_quit_clicked=lambda: None,
        )

        assert adapter.current_title() == "VoiceToText — Idle — ready"
