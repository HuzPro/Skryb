from unittest.mock import MagicMock

from voicetotext.hotkeys import keys
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.hotkeys.win32_hook_adapter import Win32HookAdapter


class TestWin32HookAdapterDelegation:
    def test_should_delegate_key_down_to_strategy(self):
        strategy = MagicMock()
        adapter = Win32HookAdapter(strategy)

        suppress = adapter._handle_key(True, keys.VK_LCONTROL)

        strategy.on_key_press.assert_called_once_with(
            keys.from_vk(keys.VK_LCONTROL),
        )
        assert suppress is False

    def test_should_delegate_key_up_to_strategy(self):
        strategy = MagicMock()
        adapter = Win32HookAdapter(strategy)

        adapter._handle_key(False, 0x4D)

        strategy.on_key_release.assert_called_once_with(keys.from_vk(0x4D))

    def test_should_never_suppress_keys(self):
        adapter = Win32HookAdapter(MagicMock())

        assert adapter._handle_key(True, 0x4D) is False
        assert adapter._handle_key(False, 0x4D) is False


class TestWin32HookAdapterPause:
    def test_should_ignore_events_when_paused(self):
        strategy = MagicMock()
        adapter = Win32HookAdapter(strategy)

        adapter.pause()
        adapter._handle_key(True, 0x4D)

        strategy.on_key_press.assert_not_called()

    def test_should_resume_processing_after_resume(self):
        strategy = MagicMock()
        adapter = Win32HookAdapter(strategy)

        adapter.pause()
        adapter.resume()
        adapter._handle_key(True, 0x4D)

        strategy.on_key_press.assert_called_once()


class TestWin32HookAdapterCallbacks:
    def test_should_wire_callbacks_to_strategy(self):
        strategy = MagicMock()
        adapter = Win32HookAdapter(strategy)
        activate = MagicMock()
        deactivate = MagicMock()

        adapter.set_on_activate(activate)
        adapter.set_on_deactivate(deactivate)

        strategy.set_callbacks.assert_called_once_with(activate, deactivate)


class TestWin32HookAdapterLifecycle:
    def test_should_start_and_stop_the_underlying_hook(self):
        adapter = Win32HookAdapter(MagicMock())
        adapter._hook = MagicMock()

        adapter.start()
        adapter.stop()

        adapter._hook.start.assert_called_once()
        adapter._hook.stop.assert_called_once()


class TestWin32HookAdapterIsAHotkeyListener:
    def test_should_implement_hotkey_listener_interface(self):
        assert isinstance(Win32HookAdapter(MagicMock()), HotkeyListener)
