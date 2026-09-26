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


class TestDeviceChanges:
    def _backend(self, changes: list[str]) -> mod.Win32TrayBackend:
        backend = mod.Win32TrayBackend.__new__(mod.Win32TrayBackend)
        backend._on_devices_changed = lambda: changes.append("changed")
        return backend

    def test_a_device_plugged_in_or_out_is_reported(self, monkeypatch) -> None:
        changes: list[str] = []
        monkeypatch.setattr(mod, "_user32", MagicMock())

        self._backend(changes)._window_proc(0, 0x0219, 0x0007, 0)

        assert changes == ["changed"]

    def test_other_device_messages_are_left_to_windows(self, monkeypatch) -> None:
        changes: list[str] = []
        user32 = MagicMock()
        monkeypatch.setattr(mod, "_user32", user32)

        self._backend(changes)._window_proc(0, 0x0219, 0x8000, 1234)

        assert changes == []
        user32.DefWindowProcW.assert_called_once_with(0, 0x0219, 0x8000, 1234)


class TestMicrophoneMenu:
    def _backend(self, connected, chosen, chosen_log):
        from voicetotext.tray import win32_tray_adapter as tray

        backend = tray.Win32TrayBackend.__new__(tray.Win32TrayBackend)
        backend._on_settings_clicked = None
        backend._on_quit_clicked = lambda: None
        backend._on_history_clicked = None
        backend._list_microphones = lambda: (connected, chosen)
        backend._on_microphone_chosen = chosen_log.append
        backend._offered_microphones = []
        return tray, backend

    def _menu_items(self, tray, backend, monkeypatch):
        items = []
        user32 = MagicMock()
        user32.CreatePopupMenu.return_value = 55
        user32.AppendMenuW.side_effect = (
            lambda menu, flags, item, label: items.append((menu, flags, item, label))
        )
        monkeypatch.setattr(tray, "_user32", user32)
        backend._append_microphone_menu(1)
        return items

    def test_lists_the_default_then_each_microphone_by_its_full_name(self, monkeypatch):
        tray, backend = self._backend(
            [("Microphone (Maono PD300X Microp", "Microphone (Maono PD300X Microphone)")],
            "",
            [],
        )

        items = self._menu_items(tray, backend, monkeypatch)

        labels = [label for menu, _flags, _item, label in items if menu == 55]
        assert labels == ["Windows default", "Microphone (Maono PD300X Microphone)"]
        assert items[-1][3] == "Microphone"

    def test_ticks_the_chosen_microphone(self, monkeypatch):
        tray, backend = self._backend([("Mic A", "Mic A"), ("Mic B", "Mic B")], "Mic B", [])

        items = self._menu_items(tray, backend, monkeypatch)

        ticked = [label for _menu, flags, _item, label in items if flags & tray._MF_CHECKED]
        assert ticked == ["Mic B"]

    def test_keeps_an_unplugged_choice_visible_and_ticked(self, monkeypatch):
        tray, backend = self._backend([("Mic A", "Mic A")], "USB Mic", [])

        items = self._menu_items(tray, backend, monkeypatch)

        ticked = [label for _menu, flags, _item, label in items if flags & tray._MF_CHECKED]
        assert ticked == ["USB Mic (not connected)"]

    def test_choosing_an_entry_saves_that_microphone(self, monkeypatch):
        chosen = []
        tray, backend = self._backend([("Mic A", "Mic A"), ("Mic B", "Mic B")], "", chosen)
        self._menu_items(tray, backend, monkeypatch)

        backend._dispatch_menu_command(tray._ID_FIRST_MICROPHONE + 2)
        backend._dispatch_menu_command(tray._ID_FIRST_MICROPHONE)

        assert chosen == ["Mic B", ""]

    def test_a_broken_device_list_still_leaves_the_rest_of_the_menu(self, monkeypatch):
        tray, backend = self._backend([], "", [])

        def broken():
            raise OSError("PortAudio not initialized")

        backend._list_microphones = broken

        assert self._menu_items(tray, backend, monkeypatch) == []


class TestRepeatLastDictationMenu:
    def _backend(self, calls):
        from voicetotext.tray import win32_tray_adapter as tray

        backend = tray.Win32TrayBackend.__new__(tray.Win32TrayBackend)
        backend._get_state_label = lambda: "Idle - ready"
        backend._on_settings_clicked = lambda: calls.append("settings")
        backend._on_quit_clicked = lambda: calls.append("quit")
        backend._on_history_clicked = lambda: calls.append("history")
        backend._on_paste_last_clicked = lambda: calls.append("paste last")
        backend._on_copy_last_clicked = lambda: calls.append("copy last")
        backend._list_microphones = None
        backend._on_microphone_chosen = None
        backend._offered_microphones = []
        backend._hwnd = 0
        return tray, backend

    def test_paste_and_copy_last_sit_above_history_in_the_menu(self, monkeypatch):
        tray, backend = self._backend([])
        labels: list[str] = []
        user32 = MagicMock()
        user32.AppendMenuW.side_effect = (
            lambda menu, flags, item, label: labels.append(label)
        )
        user32.TrackPopupMenu.return_value = 0
        monkeypatch.setattr(tray, "_user32", user32)

        backend._show_context_menu()

        items = [label for label in labels if label]
        assert items[1:4] == ["Paste last dictation", "Copy last dictation", "History..."]

    def _open_menu_and_choose(self, monkeypatch, desktop, command, calls):
        tray, backend = self._backend(calls)
        backend._hwnd = _TRAY
        backend._on_paste_last_clicked = (
            lambda: calls.append(("paste last", desktop.foreground))
        )
        desktop.choice = command
        monkeypatch.setattr(tray, "_user32", desktop)
        monkeypatch.setattr(tray, "_dwmapi", desktop)
        monkeypatch.setattr(tray, "_run_in_background", lambda work: work())
        monkeypatch.setattr(tray, "_FOCUS_RESTORE_SECONDS", 0.05)
        backend._show_context_menu()
        return tray

    def test_paste_last_hands_the_focus_back_to_the_app_the_user_was_in(
        self, monkeypatch,
    ):
        desktop = FakeDesktop([_window(_NOTEPAD, "Notepad")], foreground=_NOTEPAD)
        calls: list = []

        tray = self._open_menu_and_choose(monkeypatch, desktop, mod._ID_PASTE_LAST, calls)

        assert desktop.focus_requests == [_TRAY, _NOTEPAD]
        assert calls == [("paste last", _NOTEPAD)]
        assert tray is mod

    def test_when_the_taskbar_took_the_click_the_app_below_it_gets_the_focus(
        self, monkeypatch,
    ):
        desktop = FakeDesktop(
            [
                _window(_TASKBAR, "Shell_TrayWnd"),
                _window(0x201, "tooltips_class32", exstyle=0x80),
                _window(0x202, "Hidden", visible=False),
                _window(0x203, "Chrome_WidgetWin_1", iconic=True),
                _window(0x204, "#32770", owner=0x999),
                _window(0x205, "ApplicationFrameWindow", cloaked=True),
                _window(_NOTEPAD, "Notepad"),
                _window(0x300, "Chrome_WidgetWin_1"),
            ],
            foreground=_TASKBAR,
        )
        calls: list = []

        self._open_menu_and_choose(monkeypatch, desktop, mod._ID_PASTE_LAST, calls)

        assert desktop.focus_requests == [_TRAY, _NOTEPAD]
        assert calls == [("paste last", _NOTEPAD)]

    def test_when_the_focus_cannot_be_handed_back_paste_last_still_runs(
        self, monkeypatch,
    ):
        desktop = FakeDesktop(
            [_window(_NOTEPAD, "Notepad")], foreground=_NOTEPAD, refuses_focus=True,
        )
        calls: list = []

        self._open_menu_and_choose(monkeypatch, desktop, mod._ID_PASTE_LAST, calls)

        assert calls == [("paste last", _TRAY)]

    def test_copy_last_leaves_the_focus_alone(self, monkeypatch):
        desktop = FakeDesktop([_window(_NOTEPAD, "Notepad")], foreground=_NOTEPAD)
        calls: list = []

        self._open_menu_and_choose(monkeypatch, desktop, mod._ID_COPY_LAST, calls)

        assert desktop.focus_requests == [_TRAY]
        assert calls == ["copy last"]


_TRAY = 0x10
_TASKBAR = 0x20
_NOTEPAD = 0x100


def _window(hwnd, cls, *, visible=True, iconic=False, owner=0, exstyle=0, cloaked=False):
    return {
        "hwnd": hwnd, "cls": cls, "visible": visible, "iconic": iconic,
        "owner": owner, "exstyle": exstyle, "cloaked": cloaked,
    }


class FakeDesktop:
    """The windows on screen in Z order, and a foreground that can be refused."""

    def __init__(self, windows, *, foreground, refuses_focus=False):
        self.windows = windows
        self.foreground = foreground
        self.refuses_focus = refuses_focus
        self.focus_requests: list[int] = []
        self.choice = 0

    def _find(self, hwnd):
        return next((w for w in self.windows if w["hwnd"] == hwnd), _window(hwnd, "?"))

    def CreatePopupMenu(self):  # noqa: N802 (Win32 name)
        return 55

    def AppendMenuW(self, *_args):  # noqa: N802
        return 1

    def GetCursorPos(self, _point):  # noqa: N802
        return 1

    def TrackPopupMenu(self, *_args):  # noqa: N802
        return self.choice

    def DestroyMenu(self, _menu):  # noqa: N802
        return 1

    def GetForegroundWindow(self):  # noqa: N802
        return self.foreground

    def SetForegroundWindow(self, hwnd):  # noqa: N802
        self.focus_requests.append(hwnd)
        if hwnd == _TRAY or not self.refuses_focus:
            self.foreground = hwnd
        return 1

    def GetClassNameW(self, hwnd, buffer, _length):  # noqa: N802
        buffer.value = self._find(hwnd)["cls"]
        return len(buffer.value)

    def GetTopWindow(self, _parent):  # noqa: N802
        return self.windows[0]["hwnd"] if self.windows else 0

    def GetWindow(self, hwnd, command):  # noqa: N802
        if command == mod._GW_OWNER:
            return self._find(hwnd)["owner"]
        order = [w["hwnd"] for w in self.windows]
        position = order.index(hwnd) + 1
        return order[position] if position < len(order) else 0

    def IsWindowVisible(self, hwnd):  # noqa: N802
        return self._find(hwnd)["visible"]

    def IsIconic(self, hwnd):  # noqa: N802
        return self._find(hwnd)["iconic"]

    def GetWindowLongW(self, hwnd, _index):  # noqa: N802
        return self._find(hwnd)["exstyle"]

    def DwmGetWindowAttribute(self, hwnd, _attribute, value, _size):  # noqa: N802
        value._obj.value = 1 if self._find(hwnd)["cloaked"] else 0
        return 0
