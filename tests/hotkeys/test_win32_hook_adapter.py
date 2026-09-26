from unittest.mock import MagicMock

from voicetotext.hotkeys import keys
from voicetotext.hotkeys.actions import HotkeyActions, strategies_for
from voicetotext.hotkeys.bindings import Binding, BindingAction, BindingMode
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.win32_hook_adapter import Win32HookAdapter

_M = 0x4D
_X = 0x58
_F13 = 0x7C


class Calls:
    def __init__(self) -> None:
        self.log: list[str] = []

    def actions(self) -> HotkeyActions:
        return HotkeyActions(
            dictate_activate=lambda: self.log.append("dictate on"),
            dictate_deactivate=lambda: self.log.append("dictate off"),
            paste_last=lambda: self.log.append("paste last"),
        )


def _adapter(calls: Calls, *bindings: Binding, **kwargs) -> Win32HookAdapter:
    strategies = strategies_for(bindings, calls.actions(), double_tap_locks_hold=False)
    return Win32HookAdapter(strategies, listen_to_mouse=True, **kwargs)


def _tap(adapter: Win32HookAdapter, *vks: int) -> list[bool]:
    swallowed = [adapter._handle_key(True, vk) for vk in vks]
    swallowed += [adapter._handle_key(False, vk) for vk in reversed(vks)]
    return swallowed


class TestSeveralShortcutsInOneAdapter:
    def test_a_toggle_keyboard_shortcut_and_a_hold_mouse_shortcut_work_independently(self):
        calls = Calls()
        adapter = _adapter(
            calls,
            Binding("ctrl+alt+m", BindingMode.TOGGLE, BindingAction.DICTATE),
            Binding("mouse4", BindingMode.HOLD, BindingAction.DICTATE),
        )

        _tap(adapter, keys.VK_LCONTROL, keys.VK_LMENU, _M)
        _tap(adapter, keys.VK_LCONTROL, keys.VK_LMENU, _M)
        adapter._handle_key(True, keys.VK_XBUTTON1)
        for _ in range(3):
            _tap(adapter, _X)
        adapter._handle_key(False, keys.VK_XBUTTON1)

        assert calls.log == ["dictate on", "dictate on", "dictate on", "dictate off"]

    def test_a_shortcut_with_no_action_yet_is_inert(self):
        calls = Calls()
        adapter = _adapter(
            calls,
            Binding("ctrl+alt+m", BindingMode.TOGGLE, BindingAction.DICTATE),
            Binding("mouse5", BindingMode.PRESS, BindingAction.COPY_LAST),
            Binding("ctrl+alt+e", BindingMode.HOLD, BindingAction.EDIT_SELECTION),
        )

        assert _tap(adapter, keys.VK_XBUTTON2) == [False, False]
        _tap(adapter, keys.VK_LCONTROL, keys.VK_LMENU, 0x45)

        assert calls.log == []

    def test_a_single_press_shortcut_fires_once_per_press(self):
        calls = Calls()
        adapter = _adapter(
            calls,
            Binding("f12", BindingMode.TOGGLE, BindingAction.DICTATE),
            Binding("ctrl+f13", BindingMode.PRESS, BindingAction.PASTE_LAST),
        )

        adapter._handle_key(True, keys.VK_LCONTROL)
        for _ in range(2):
            _tap(adapter, _F13)
        adapter._handle_key(False, keys.VK_LCONTROL)

        assert calls.log == ["paste last", "paste last"]


class TestMouseButtonsAreSwallowedOnlyWhenTheyCompleteAShortcut:
    def test_a_bound_button_is_swallowed_down_and_up_and_an_unbound_one_passes(self):
        calls = Calls()
        adapter = _adapter(
            calls, Binding("mouse4", BindingMode.HOLD, BindingAction.DICTATE),
        )

        assert _tap(adapter, keys.VK_XBUTTON1) == [True, True]
        assert _tap(adapter, keys.VK_XBUTTON2) == [False, False]
        assert _tap(adapter, keys.VK_MBUTTON) == [False, False]

    def test_a_button_pressed_without_its_modifier_reaches_the_app(self):
        calls = Calls()
        adapter = _adapter(
            calls, Binding("ctrl+mouse4", BindingMode.TOGGLE, BindingAction.DICTATE),
        )

        assert _tap(adapter, keys.VK_XBUTTON1) == [False, False]
        assert _tap(adapter, keys.VK_LCONTROL, keys.VK_XBUTTON1) == [
            False, True, True, False,
        ]
        assert calls.log == ["dictate on"]

    def test_keyboard_shortcut_keys_always_reach_the_app(self):
        calls = Calls()
        adapter = _adapter(
            calls, Binding("ctrl+alt+m", BindingMode.TOGGLE, BindingAction.DICTATE),
        )

        assert _tap(adapter, keys.VK_LCONTROL, keys.VK_LMENU, _M) == [False] * 6
        assert calls.log == ["dictate on"]


class TestEscapeCancelsARecording:
    def test_escape_cancels_and_is_swallowed_while_recording(self):
        strategy = MagicMock()
        cancel = MagicMock(return_value=True)
        adapter = Win32HookAdapter([strategy], cancel=cancel)

        assert adapter._handle_key(True, keys.VK_ESCAPE) is True
        assert adapter._handle_key(False, keys.VK_ESCAPE) is True
        cancel.assert_called_once()
        strategy.on_key_press.assert_not_called()

    def test_escape_reaches_the_app_when_nothing_is_recording(self):
        adapter = Win32HookAdapter([], cancel=lambda: False)

        assert adapter._handle_key(True, keys.VK_ESCAPE) is False
        assert adapter._handle_key(False, keys.VK_ESCAPE) is False

    def test_a_held_escape_cancels_once_and_stays_swallowed_until_released(self):
        cancel = MagicMock(return_value=True)
        adapter = Win32HookAdapter([], cancel=cancel)

        repeats = [adapter._handle_key(True, keys.VK_ESCAPE) for _ in range(30)]
        released = adapter._handle_key(False, keys.VK_ESCAPE)

        assert all(repeats) and released
        cancel.assert_called_once()

    def test_the_next_escape_after_a_cancel_reaches_the_app(self):
        results = iter([True, False])
        adapter = Win32HookAdapter([], cancel=lambda: next(results))
        adapter._handle_key(True, keys.VK_ESCAPE)
        adapter._handle_key(False, keys.VK_ESCAPE)

        assert adapter._handle_key(True, keys.VK_ESCAPE) is False
        assert adapter._handle_key(False, keys.VK_ESCAPE) is False

    def test_other_keys_are_never_swallowed_while_recording(self):
        adapter = Win32HookAdapter([], cancel=lambda: True)

        assert adapter._handle_key(True, _M) is False


class TestWin32HookAdapterPause:
    def test_a_paused_adapter_ignores_every_shortcut_until_resumed(self):
        calls = Calls()
        adapter = _adapter(
            calls, Binding("mouse4", BindingMode.TOGGLE, BindingAction.DICTATE),
        )

        adapter.pause()
        paused = _tap(adapter, keys.VK_XBUTTON1)
        adapter.resume()
        resumed = _tap(adapter, keys.VK_XBUTTON1)

        assert paused == [False, False]
        assert resumed == [True, True]
        assert calls.log == ["dictate on"]


class TestWin32HookAdapterLifecycle:
    def test_should_start_and_stop_both_hooks_when_a_shortcut_uses_the_mouse(self):
        adapter = Win32HookAdapter([], listen_to_mouse=True)
        adapter._hook = MagicMock()
        adapter._mouse_hook = MagicMock()

        adapter.start()
        adapter.stop()

        adapter._hook.start.assert_called_once()
        adapter._hook.stop.assert_called_once()
        adapter._mouse_hook.start.assert_called_once()
        adapter._mouse_hook.stop.assert_called_once()

    def test_no_mouse_hook_is_installed_when_no_shortcut_uses_the_mouse(self):
        adapter = Win32HookAdapter([])
        adapter._hook = MagicMock()

        adapter.start()
        adapter.stop()

        assert adapter._mouse_hook is None


class TestWin32HookAdapterIsAHotkeyListener:
    def test_should_implement_hotkey_listener_interface(self):
        assert isinstance(Win32HookAdapter([]), HotkeyListener)
