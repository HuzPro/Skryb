"""Tests for the native tray adapter's orchestration.

The :class:`Win32TrayBackend` (message window, ``Shell_NotifyIcon``, menu)
needs a live shell and is covered by the manual Windows smoke test. Here
the orchestration is driven with a fake backend and a fake dispatcher;
``create_hicon`` is exercised against real GDI (this module is win32-only).
"""

from unittest.mock import MagicMock

import pytest

from voicetotext.core.state_machine import AppState
from voicetotext.tray import win32_tray_adapter as mod
from voicetotext.tray.win32_tray_adapter import (
    Win32TrayAdapter,
    build_win32_tray_adapter,
    create_hicon,
)


class SyncDispatcher:
    def dispatch(self, action) -> None:
        action()


class RecordingDispatcher:
    def __init__(self) -> None:
        self.actions: list = []

    def dispatch(self, action) -> None:
        self.actions.append(action)


def _make_adapter(*, root, dispatcher, backend, label="Idle - ready"):
    return Win32TrayAdapter(
        initial_state=AppState.IDLE,
        initial_label=label,
        get_state_label=lambda: label,
        on_settings_clicked=lambda: None,
        on_quit_clicked=lambda: None,
        root=root,
        dispatcher=dispatcher,
        backend=backend,
    )


@pytest.fixture
def backend():
    return MagicMock()


@pytest.fixture
def adapter(tk_root, backend):
    a = _make_adapter(root=tk_root, dispatcher=SyncDispatcher(), backend=backend)
    yield a
    a.set_elapsed_provider(None)  # cancel any pending after-timer


class TestCreateHicon:
    def test_should_build_a_non_null_icon_handle(self):
        hicon = create_hicon(AppState.RECORDING)

        assert hicon != 0

        mod._user32.DestroyIcon(hicon)


class TestInitial:
    def test_should_set_icon_on_backend(self, adapter, backend):
        assert backend.set_icon.call_count == 1

    def test_should_set_tooltip_with_title_prefix(self, adapter, backend):
        backend.set_tooltip.assert_called_with("VoiceToText - Idle - ready")


class TestVisible:
    def test_should_start_invisible(self, adapter):
        assert adapter.visible is False

    def test_should_show_backend_when_set_visible_true(self, adapter, backend):
        adapter.visible = True

        assert adapter.visible is True
        backend.show.assert_called()

    def test_should_hide_backend_when_set_visible_false(self, adapter, backend):
        adapter.visible = True
        adapter.visible = False

        backend.hide.assert_called()


class TestUpdateState:
    def test_should_set_new_icon_and_tooltip(self, adapter, backend):
        backend.reset_mock()

        adapter.update_state(AppState.RECORDING, "Recording...")

        backend.set_icon.assert_called_once()
        backend.set_tooltip.assert_called_with("VoiceToText - Recording...")

    def test_should_marshal_state_update_through_dispatcher(self, tk_root, backend):
        dispatcher = RecordingDispatcher()
        adapter = _make_adapter(
            root=tk_root, dispatcher=dispatcher, backend=backend,
        )
        backend.reset_mock()

        adapter.update_state(AppState.RECORDING, "Recording...")

        backend.set_icon.assert_not_called()  # deferred
        dispatcher.actions[0]()
        backend.set_icon.assert_called_once()


class TestNotify:
    def test_should_forward_notification_to_backend(self, adapter, backend):
        adapter.notify("VoiceToText", "Clipboard write failed.")

        backend.notify.assert_called_once_with(
            "VoiceToText", "Clipboard write failed.",
        )


class TestElapsedProvider:
    def test_should_render_elapsed_in_tooltip(self, adapter, backend):
        adapter.set_elapsed_provider(lambda: 47.9)
        backend.reset_mock()

        adapter._refresh_elapsed_tooltip()

        backend.set_tooltip.assert_called_with("VoiceToText - Recording 0:47")

    def test_should_restore_static_tooltip_when_cleared(self, adapter, backend):
        adapter.set_elapsed_provider(lambda: 10.0)
        backend.reset_mock()

        adapter.set_elapsed_provider(None)

        backend.set_tooltip.assert_called_with("VoiceToText - Idle - ready")

    def test_should_no_op_refresh_after_provider_cleared(self, adapter, backend):
        adapter.set_elapsed_provider(lambda: 99.0)
        adapter.set_elapsed_provider(None)
        backend.reset_mock()

        adapter._refresh_elapsed_tooltip()

        backend.set_tooltip.assert_not_called()


class TestRunAndStop:
    def test_should_invoke_setup_with_self_then_run_loop(self, backend):
        root = MagicMock()
        adapter = _make_adapter(
            root=root, dispatcher=SyncDispatcher(), backend=backend,
        )
        received: list = []

        adapter.run(received.append)

        assert received == [adapter]
        root.mainloop.assert_called_once()

    def test_should_destroy_backend_and_quit_loop_on_stop(self, backend):
        root = MagicMock()
        adapter = _make_adapter(
            root=root, dispatcher=SyncDispatcher(), backend=backend,
        )

        adapter.stop()

        backend.destroy.assert_called_once()
        root.quit.assert_called_once()


class TestFactory:
    def test_should_build_adapter_with_state_label_title(self, tk_root):
        dispatcher = SyncDispatcher()
        adapter = build_win32_tray_adapter(
            initial_state=AppState.IDLE,
            get_state_label=lambda: "Idle - ready",
            on_settings_clicked=None,
            on_quit_clicked=lambda: None,
            root=tk_root,
            dispatcher=dispatcher,
        )

        assert adapter.current_title() == "VoiceToText - Idle - ready"

        adapter.stop()  # tears down the real backend window


class TestExplorerRestartRecovery:
    """Explorer restarting silently drops every tray icon it was hosting."""

    def _backend(self, *, added: bool) -> mod.Win32TrayBackend:
        backend = mod.Win32TrayBackend.__new__(mod.Win32TrayBackend)
        backend._added = added
        backend._nid = mod._NOTIFYICONDATAW()
        return backend

    def test_should_readd_a_shown_icon_when_the_shell_says_taskbar_created(
        self, monkeypatch,
    ) -> None:
        backend = self._backend(added=True)
        shell = MagicMock()
        monkeypatch.setattr(mod, "_shell32", shell)

        backend._window_proc(0, mod._WM_TASKBAR_CREATED, 0, 0)

        assert shell.Shell_NotifyIconW.call_args.args[0] == mod._NIM_ADD
        assert backend._added is True

    def test_should_leave_a_hidden_icon_hidden_when_taskbar_created(
        self, monkeypatch,
    ) -> None:
        backend = self._backend(added=False)
        shell = MagicMock()
        monkeypatch.setattr(mod, "_shell32", shell)

        backend._window_proc(0, mod._WM_TASKBAR_CREATED, 0, 0)

        shell.Shell_NotifyIconW.assert_not_called()
