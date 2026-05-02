"""Tests for ``PynputCaptureBridge`` — the suppressing key-capture
bridge used during hotkey recording.

The bridge owns a pynput ``Listener`` configured with ``suppress=True``
so OS-level shortcuts cannot fire while a hotkey is being recorded.
Each pressed key is converted to a canonical name and emitted onto the
GUI thread via a Qt signal.

These tests inject a fake listener factory so no real keyboard hook
is installed.
"""

from unittest.mock import MagicMock

from pynput.keyboard import Key, KeyCode

from voicetotext.settings.pynput_capture_bridge import PynputCaptureBridge


def _make_bridge():
    factory = MagicMock()
    fake_listener = MagicMock()
    factory.return_value = fake_listener
    bridge = PynputCaptureBridge(listener_factory=factory)
    return bridge, factory, fake_listener


class TestPynputCaptureBridgeStart:
    def test_should_construct_listener_with_suppress_true(self, qtbot):
        bridge, factory, _ = _make_bridge()

        bridge.start()

        kwargs = factory.call_args.kwargs
        assert kwargs["suppress"] is True
        assert callable(kwargs["on_press"])

    def test_should_start_listener(self, qtbot):
        bridge, _, fake_listener = _make_bridge()

        bridge.start()

        fake_listener.start.assert_called_once()

    def test_should_be_idempotent_when_started_twice(self, qtbot):
        bridge, factory, _ = _make_bridge()

        bridge.start()
        bridge.start()

        assert factory.call_count == 1


class TestPynputCaptureBridgeStop:
    def test_should_stop_listener(self, qtbot):
        bridge, _, fake_listener = _make_bridge()
        bridge.start()

        bridge.stop()

        fake_listener.stop.assert_called_once()

    def test_should_be_safe_to_stop_when_never_started(self, qtbot):
        bridge, _, fake_listener = _make_bridge()

        bridge.stop()

        fake_listener.stop.assert_not_called()

    def test_should_allow_restart_after_stop(self, qtbot):
        bridge, factory, _ = _make_bridge()
        bridge.start()
        bridge.stop()

        bridge.start()

        assert factory.call_count == 2


class TestPynputCaptureBridgeSignal:
    def test_should_emit_canonical_name_on_modifier_press(self, qtbot):
        bridge, factory, _ = _make_bridge()
        bridge.start()
        on_press = factory.call_args.kwargs["on_press"]
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)

        on_press(Key.ctrl_l)

        assert emitted == ["ctrl"]

    def test_should_emit_canonical_name_on_letter_press(self, qtbot):
        bridge, factory, _ = _make_bridge()
        bridge.start()
        on_press = factory.call_args.kwargs["on_press"]
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)

        on_press(KeyCode.from_char("m"))

        assert emitted == ["m"]

    def test_should_not_emit_when_key_is_unrecognized(self, qtbot):
        bridge, factory, _ = _make_bridge()
        bridge.start()
        on_press = factory.call_args.kwargs["on_press"]
        emitted: list[str] = []
        bridge.key_pressed.connect(emitted.append)

        on_press(Key.print_screen)

        assert emitted == []
