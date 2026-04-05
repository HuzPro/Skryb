from unittest.mock import MagicMock, patch

from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.pynput_adapter import PynputAdapter


class TestPynputAdapterStart:
    @patch("voicetotext.hotkeys.pynput_adapter.keyboard")
    def test_should_create_and_start_listener(self, mock_kb):
        strategy = MagicMock()
        adapter = PynputAdapter(strategy)

        adapter.start()

        mock_kb.Listener.assert_called_once()
        mock_kb.Listener.return_value.start.assert_called_once()


class TestPynputAdapterStop:
    @patch("voicetotext.hotkeys.pynput_adapter.keyboard")
    def test_should_stop_listener(self, mock_kb):
        strategy = MagicMock()
        adapter = PynputAdapter(strategy)
        adapter.start()

        adapter.stop()

        mock_kb.Listener.return_value.stop.assert_called_once()


class TestPynputAdapterDelegation:
    @patch("voicetotext.hotkeys.pynput_adapter.keyboard")
    def test_should_delegate_key_events_to_strategy(self, mock_kb):
        strategy = MagicMock()
        adapter = PynputAdapter(strategy)
        adapter.start()

        listener_kwargs = mock_kb.Listener.call_args[1]
        on_press = listener_kwargs["on_press"]
        on_release = listener_kwargs["on_release"]

        fake_key = MagicMock()
        on_press(fake_key)
        on_release(fake_key)

        strategy.on_key_press.assert_called_once_with(fake_key)
        strategy.on_key_release.assert_called_once_with(fake_key)


class TestPynputAdapterCallbacks:
    @patch("voicetotext.hotkeys.pynput_adapter.keyboard")
    def test_should_wire_callbacks_to_strategy(self, mock_kb):
        strategy = MagicMock()
        adapter = PynputAdapter(strategy)
        activate_cb = MagicMock()
        deactivate_cb = MagicMock()

        adapter.set_on_activate(activate_cb)
        adapter.set_on_deactivate(deactivate_cb)

        strategy.set_callbacks.assert_called_once_with(activate_cb, deactivate_cb)


class TestPynputAdapterIsAHotkeyListener:
    @patch("voicetotext.hotkeys.pynput_adapter.keyboard")
    def test_should_implement_hotkey_listener_interface(self, mock_kb):
        adapter = PynputAdapter(MagicMock())

        assert isinstance(adapter, HotkeyListener)
