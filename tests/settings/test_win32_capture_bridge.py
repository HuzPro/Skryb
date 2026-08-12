"""Tests for ``Win32CaptureBridge`` - the suppressing key-capture bridge
used during hotkey recording.

The bridge installs a native low-level keyboard hook with suppression so
OS shortcuts cannot fire while recording, converts each pressed key to a
canonical name, and emits it through a callback-list ``Signal``.
"""

from unittest.mock import MagicMock

from voicetotext.hotkeys import keys
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


class TestWin32CaptureBridgeLifecycle:
    def test_should_start_and_stop_the_underlying_hook(self):
        bridge = Win32CaptureBridge()
        bridge._hook = MagicMock()

        bridge.start()
        bridge.stop()

        bridge._hook.start.assert_called_once()
        bridge._hook.stop.assert_called_once()
