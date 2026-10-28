import ctypes
import random
import sys
from unittest.mock import MagicMock

import pytest

from skryb.core.exceptions import HotkeyError, OutputError
from skryb.hotkeys import keys
from skryb.hotkeys import win32_keyboard_hook as wkh
from skryb.hotkeys.actions import HotkeyActions, strategies_for
from skryb.hotkeys.bindings import Binding, BindingAction, BindingMode
from skryb.hotkeys.listener import HotkeyListener
from skryb.hotkeys.win32_hook_adapter import Win32HookAdapter

_C = 0x43
_M = 0x4D
_X = 0x58
_F13 = 0x7C
_F23 = 0x86
_QUOTE = 0xDE


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


def _dictate(combo: str, mode: BindingMode = BindingMode.TOGGLE) -> Binding:
    return Binding(combo, mode, BindingAction.DICTATE)


def _press(adapter: Win32HookAdapter, *vks: int) -> list[bool]:
    return [adapter._handle_key(True, vk) for vk in vks]


def _release(adapter: Win32HookAdapter, *vks: int) -> list[bool]:
    return [adapter._handle_key(False, vk) for vk in vks]


def _tap(adapter: Win32HookAdapter, *vks: int) -> list[bool]:
    return _press(adapter, *vks) + _release(adapter, *reversed(vks))


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
        )

        assert _tap(adapter, keys.VK_XBUTTON2) == [False, False]

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


class TestTheKeyThatCompletesAKeyboardShortcutIsSwallowed:
    def test_the_last_key_is_swallowed_down_and_up_and_the_modifiers_reach_the_app(self):
        calls = Calls()
        adapter = _adapter(calls, _dictate("ctrl+alt+m"))

        assert _tap(adapter, keys.VK_LCONTROL, keys.VK_LMENU, _M) == [
            False, False, True, True, False, False,
        ]
        assert calls.log == ["dictate on"]

    def test_keys_that_do_not_complete_the_shortcut_reach_the_app(self):
        calls = Calls()
        adapter = _adapter(calls, _dictate("ctrl+alt+m"))

        assert _tap(adapter, _M) == [False, False]
        assert _tap(adapter, keys.VK_LCONTROL, _M) == [False] * 4
        assert _tap(adapter, keys.VK_LMENU, keys.VK_LCONTROL, _M) == [False] * 6
        assert calls.log == []

    def test_a_held_last_key_stays_swallowed_through_every_repeat_until_released(self):
        calls = Calls()
        adapter = _adapter(calls, _dictate("ctrl+alt+m", BindingMode.HOLD))

        _press(adapter, keys.VK_LCONTROL, keys.VK_LMENU)
        repeats = _press(adapter, *[_M] * 200)
        released = _release(adapter, _M, keys.VK_LMENU, keys.VK_LCONTROL)

        assert all(repeats)
        assert released == [True, False, False]
        assert calls.log == ["dictate on", "dictate off"]

    def test_the_next_press_of_a_swallowed_key_on_its_own_reaches_the_app(self):
        adapter = _adapter(Calls(), _dictate("ctrl+alt+m"))
        _tap(adapter, keys.VK_LCONTROL, keys.VK_LMENU, _M)

        assert _tap(adapter, _M) == [False, False]

    def test_copilot_then_quote_dictates_while_copilot_alone_and_copilot_c_reach_windows(self):
        calls = Calls()
        adapter = _adapter(calls, _dictate("cmd+shift+f23+quote"))
        copilot = (keys.VK_LWIN, keys.VK_LSHIFT, _F23)

        alone = _tap(adapter, *copilot)
        copy = _tap(adapter, *copilot, _C)
        dictate = _tap(adapter, *copilot, _QUOTE)

        assert alone == [False] * 6
        assert copy == [False] * 8
        assert dictate == [False, False, False, True, True, False, False, False]
        assert calls.log == ["dictate on"]

    def test_a_key_swallowed_before_a_pause_does_not_swallow_its_next_press(self):
        adapter = _adapter(Calls(), _dictate("ctrl+alt+m"))
        _press(adapter, keys.VK_LCONTROL, keys.VK_LMENU, _M)

        adapter.pause()
        _release(adapter, _M, keys.VK_LMENU, keys.VK_LCONTROL)
        adapter.resume()

        assert _tap(adapter, _M) == [False, False]


# Copilot + ' on a Dell XPS 13 whose Copilot key is set to act as Right Ctrl,
# as a low-level hook saw it on 2026-10-21. Windows releases Win+Shift+F23 at
# once and holds Right Ctrl (0xA3) instead; its release comes back injected.
_COPILOT_QUOTE_UNDER_RIGHT_CTRL_SWAP = [
    (True, keys.VK_LWIN), (True, keys.VK_LSHIFT), (True, _F23),
    (False, keys.VK_LWIN), (False, keys.VK_LSHIFT), (False, _F23),
    (True, keys.VK_RCONTROL), (True, _QUOTE), (False, _QUOTE),
    (False, _F23), (False, keys.VK_RCONTROL), (False, keys.VK_LSHIFT), (False, keys.VK_LWIN),
]


class TestTheCopilotKeySwappedForRightCtrl:
    def test_right_ctrl_quote_fires_on_each_press_and_swallows_only_the_quote(self):
        calls = Calls()
        masks: list[str] = []
        adapter = _adapter(
            calls, _dictate("ctrl_r+quote"), mask_menus=lambda: masks.append("mask"),
        )
        events = _COPILOT_QUOTE_UNDER_RIGHT_CTRL_SWAP * 2

        swallowed = [adapter._handle_key(is_down, vk) for is_down, vk in events]

        assert calls.log == ["dictate on", "dictate on"]
        assert [event for event, gone in zip(events, swallowed) if gone] == [
            (True, _QUOTE), (False, _QUOTE),
        ] * 2
        assert masks == []

    def test_the_raw_copilot_keys_never_match_because_windows_releases_them_at_once(self):
        calls = Calls()
        adapter = _adapter(calls, _dictate("cmd+shift+f23+quote"))

        swallowed = [
            adapter._handle_key(is_down, vk)
            for is_down, vk in _COPILOT_QUOTE_UNDER_RIGHT_CTRL_SWAP
        ]

        assert not any(swallowed)
        assert calls.log == []


class TestWinAndAltDoNotOpenMenusWhenTheShortcutKeyIsSwallowed:
    def _adapter_counting_masks(self, combo: str, mode: BindingMode = BindingMode.TOGGLE):
        masks: list[str] = []
        adapter = _adapter(Calls(), _dictate(combo, mode), mask_menus=lambda: masks.append("mask"))
        return adapter, masks

    @pytest.mark.parametrize(("combo", "pressed"), [
        ("ctrl+alt+m", (keys.VK_LCONTROL, keys.VK_LMENU, _M)),
        ("alt_r+x", (keys.VK_RMENU, _X)),
        ("cmd+shift+f23+quote", (keys.VK_LWIN, keys.VK_LSHIFT, _F23, _QUOTE)),
        ("cmd+mouse4", (keys.VK_LWIN, keys.VK_XBUTTON1)),
    ])
    def test_one_mask_key_is_sent_when_win_or_alt_is_held(self, combo, pressed):
        adapter, masks = self._adapter_counting_masks(combo)

        _tap(adapter, *pressed)

        assert masks == ["mask"]

    def test_no_mask_key_is_sent_when_only_ctrl_and_shift_are_held(self):
        adapter, masks = self._adapter_counting_masks("ctrl+shift+m")

        _tap(adapter, keys.VK_LCONTROL, keys.VK_LSHIFT, _M)

        assert masks == []

    def test_no_mask_key_is_sent_for_keys_that_reach_the_app(self):
        adapter, masks = self._adapter_counting_masks("ctrl+alt+m")

        _tap(adapter, keys.VK_LMENU, _X)
        _tap(adapter, keys.VK_LWIN, _M)

        assert masks == []

    def test_a_held_shortcut_key_sends_one_mask_key_however_long_it_repeats(self):
        adapter, masks = self._adapter_counting_masks("ctrl+alt+m", BindingMode.HOLD)

        _press(adapter, keys.VK_LCONTROL, keys.VK_LMENU, *[_M] * 100)
        _release(adapter, _M, keys.VK_LMENU, keys.VK_LCONTROL)

        assert masks == ["mask"]

    def test_a_refused_mask_key_still_swallows_the_shortcut_key(self):
        def refuse() -> None:
            raise OutputError("SendInput injected 0 of 2 menu mask key events.")

        calls = Calls()
        adapter = _adapter(calls, _dictate("ctrl+alt+m"), mask_menus=refuse)

        assert _tap(adapter, keys.VK_LCONTROL, keys.VK_LMENU, _M) == [
            False, False, True, True, False, False,
        ]
        assert calls.log == ["dictate on"]


class TestALongRandomTypingSession:
    def test_five_thousand_random_key_events_swallow_exactly_the_keys_that_fired_a_shortcut(self):
        rng = random.Random(20261021)
        calls = Calls()
        masks: list[str] = []
        adapter = _adapter(
            calls,
            _dictate("alt+m"),
            _dictate("mouse4", BindingMode.HOLD),
            Binding("ctrl+f13", BindingMode.PRESS, BindingAction.PASTE_LAST),
            mask_menus=lambda: masks.append("mask"),
        )
        pool = [
            keys.VK_LCONTROL, keys.VK_LMENU, keys.VK_RMENU, keys.VK_LSHIFT, keys.VK_LWIN,
            _M, _X, _QUOTE, _F13, _F23, keys.VK_XBUTTON1,
        ]
        held: set[int] = set()
        swallowed_presses: set[int] = set()
        fresh_swallows = 0
        last_pressed = 0
        events = 0

        while events < 5000:
            if held and rng.random() < 0.45:
                vk = rng.choice(sorted(held))
                assert adapter._handle_key(False, vk) == (vk in swallowed_presses)
                held.discard(vk)
                swallowed_presses.discard(vk)
                events += 1
                continue
            vk = rng.choice(pool)
            # Windows repeats only the key pressed last, and never a mouse button.
            if vk in held and (vk != last_pressed or vk == keys.VK_XBUTTON1):
                continue
            swallowed = adapter._handle_key(True, vk)
            last_pressed = vk
            events += 1
            if vk in held:
                assert swallowed == (vk in swallowed_presses)
                continue
            held.add(vk)
            if swallowed:
                swallowed_presses.add(vk)
                fresh_swallows += 1
        for vk in sorted(held):
            adapter._handle_key(False, vk)

        fired = [entry for entry in calls.log if entry != "dictate off"]
        assert fresh_swallows == len(fired) > 100
        assert 0 < len(masks) <= fresh_swallows
        assert adapter._swallowed == set()


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

    def test_the_mouse_hook_is_removed_even_when_the_keyboard_hook_thread_is_stuck(self):
        adapter = Win32HookAdapter([], listen_to_mouse=True)
        adapter._hook = MagicMock()
        adapter._hook.stop.side_effect = HotkeyError("keyboard hook thread did not exit")
        adapter._mouse_hook = MagicMock()

        with pytest.raises(HotkeyError):
            adapter.stop()

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


class TestInjectedTextThroughTheRealHookProc:
    """Events fed to the keyboard hook's ctypes thunk, as Windows would."""

    def _press(self, adapter: Win32HookAdapter, message: int, vk: int, scan_code: int = 0) -> int:
        info = wkh._KBDLLHOOKSTRUCT(vkCode=vk, scanCode=scan_code, flags=0x10)
        return adapter._hook._proc(wkh._HC_ACTION, message, ctypes.addressof(info))

    def test_text_injected_in_the_middle_of_a_shortcut_does_not_break_it(self, monkeypatch):
        reported: list[object] = []
        monkeypatch.setattr(sys, "unraisablehook", reported.append)
        user32 = MagicMock()
        user32.CallNextHookEx.return_value = 0
        monkeypatch.setattr(wkh, "_user32", user32)
        calls = Calls()
        adapter = _adapter(
            calls, Binding("ctrl+alt+m", BindingMode.TOGGLE, BindingAction.DICTATE),
        )

        self._press(adapter, wkh._WM_KEYDOWN, keys.VK_LCONTROL)
        for unit in memoryview("héllo 😀".encode("utf-16-le")).cast("H"):
            self._press(adapter, wkh._WM_KEYDOWN, 0xE7, unit)
            self._press(adapter, wkh._WM_KEYUP, 0xE7, unit)
        self._press(adapter, wkh._WM_KEYDOWN, 0)
        self._press(adapter, wkh._WM_SYSKEYDOWN, keys.VK_LMENU)
        self._press(adapter, wkh._WM_SYSKEYDOWN, _M)

        assert calls.log == ["dictate on"]
        assert reported == []
