"""Tests for ``Win32CaptureBridge`` - the suppressing key-capture bridge
used during hotkey recording.

The bridge installs a native low-level keyboard hook with suppression so
OS shortcuts cannot fire while recording, converts each pressed key to a
canonical name, and emits it through a callback-list ``Signal``.
"""

import ctypes
import sys
from unittest.mock import MagicMock

import pytest

from voicetotext.core.exceptions import HotkeyError
from voicetotext.hotkeys import keys
from voicetotext.hotkeys import win32_keyboard_hook as wkh
from voicetotext.hotkeys import win32_mouse_hook as wmh
from voicetotext.settings.win32_capture_bridge import Win32CaptureBridge


class TestWin32CaptureBridgeSignal:
    def test_should_emit_canonical_name_on_modifier_press(self):
        bridge = Win32CaptureBridge()
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)

        bridge._handle_key(True, keys.VK_LCONTROL)

        assert emitted == ["ctrl"]

    def test_should_emit_canonical_name_on_letter_press(self):
        bridge = Win32CaptureBridge()
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)

        bridge._handle_key(True, 0x4D)

        assert emitted == ["m"]

    def test_should_not_emit_on_key_release(self):
        bridge = Win32CaptureBridge()
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)

        bridge._handle_key(False, keys.VK_LCONTROL)

        assert emitted == []

    def test_should_not_emit_for_unrecognized_key(self):
        bridge = Win32CaptureBridge()
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)

        bridge._handle_key(True, 0x2C)  # Print Screen

        assert emitted == []


class TestWin32CaptureBridgeSuppression:
    def test_should_suppress_every_key_while_recording(self):
        bridge = Win32CaptureBridge()

        assert bridge._handle_key(True, 0x4D) is True
        assert bridge._handle_key(False, 0x4D) is True
        assert bridge._handle_key(True, 0x2C) is True

    @pytest.mark.parametrize(("vk", "name"), [
        (keys.VK_XBUTTON1, "mouse4"),
        (keys.VK_XBUTTON2, "mouse5"),
        (keys.VK_MBUTTON, "mouse_middle"),
    ])
    def test_a_side_or_middle_button_is_recorded_and_swallowed(self, vk, name):
        bridge = Win32CaptureBridge()
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)

        assert bridge._handle_key(True, vk) is True
        assert bridge._handle_key(False, vk) is True
        assert emitted == [name]


def _bridge_with_fake_hooks() -> Win32CaptureBridge:
    bridge = Win32CaptureBridge()
    bridge._hook = MagicMock()
    bridge._mouse_hook = MagicMock()
    return bridge


class TestWin32CaptureBridgeLifecycle:
    def test_should_start_and_stop_the_keyboard_and_mouse_hooks(self):
        bridge = _bridge_with_fake_hooks()

        bridge.start()
        bridge.stop()

        bridge._hook.start.assert_called_once()
        bridge._hook.stop.assert_called_once()
        bridge._mouse_hook.start.assert_called_once()
        bridge._mouse_hook.stop.assert_called_once()

    def test_the_hooks_stay_until_every_capture_sharing_them_has_stopped(self):
        bridge = _bridge_with_fake_hooks()

        bridge.start()
        bridge.start()
        bridge.stop()

        bridge._hook.stop.assert_not_called()

        bridge.stop()
        bridge.stop()

        bridge._hook.start.assert_called_once()
        bridge._hook.stop.assert_called_once()
        bridge._mouse_hook.stop.assert_called_once()

    def test_the_mouse_hook_is_removed_even_when_the_keyboard_hook_thread_is_stuck(self):
        bridge = _bridge_with_fake_hooks()
        bridge._hook.stop.side_effect = HotkeyError("keyboard hook thread did not exit")
        bridge.start()

        with pytest.raises(HotkeyError):
            bridge.stop()

        bridge._mouse_hook.stop.assert_called_once()


class TestBothCaptureHooksThroughTheirRealProcs:
    """Odd events fed to each of the bridge's ctypes thunks, as Windows would."""

    @pytest.fixture
    def next_hook(self, monkeypatch) -> MagicMock:
        user32 = MagicMock()
        user32.CallNextHookEx.return_value = 5
        monkeypatch.setattr(wkh, "_user32", user32)
        return user32.CallNextHookEx

    @pytest.fixture
    def reported(self, monkeypatch) -> list[object]:
        escaped: list[object] = []
        monkeypatch.setattr(sys, "unraisablehook", escaped.append)
        return escaped

    @pytest.mark.parametrize(("vk", "scan_code"), [
        (0xE7, ord("m")), (0xE7, 0xD83D), (0, 0x32), (0xFF, 0x73),
    ])
    def test_injected_text_and_scan_code_only_keys_are_neither_recorded_nor_swallowed(
        self, next_hook, reported, vk, scan_code,
    ):
        bridge = Win32CaptureBridge()
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)
        info = wkh._KBDLLHOOKSTRUCT(vkCode=vk, scanCode=scan_code, flags=0x12)

        result = bridge._hook._proc(wkh._HC_ACTION, wkh._WM_KEYDOWN, ctypes.addressof(info))

        assert result == 5
        assert emitted == []
        assert reported == []

    def test_a_real_key_is_still_recorded_and_swallowed(self, next_hook, reported):
        bridge = Win32CaptureBridge()
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)
        info = wkh._KBDLLHOOKSTRUCT(vkCode=0x78, flags=0x10)

        result = bridge._hook._proc(wkh._HC_ACTION, wkh._WM_KEYDOWN, ctypes.addressof(info))

        assert result == wkh._SUPPRESS
        assert emitted == ["f9"]
        assert reported == []

    @pytest.mark.parametrize(("hook_name", "message"), [
        ("_hook", wkh._WM_KEYDOWN),
        ("_hook", wkh._WM_SYSKEYUP),
        ("_mouse_hook", wmh._WM_XBUTTONDOWN),
        ("_mouse_hook", wmh._WM_MBUTTONUP),
    ])
    def test_an_event_without_its_data_is_passed_on_by_either_hook(
        self, next_hook, reported, hook_name, message,
    ):
        bridge = Win32CaptureBridge()
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)

        result = getattr(bridge, hook_name)._proc(wkh._HC_ACTION, message, 0)

        assert result == 5
        assert emitted == []
        assert reported == []
