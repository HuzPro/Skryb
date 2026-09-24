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


class TestEscapeCancelsARecording:
    def test_escape_cancels_and_is_swallowed_while_recording(self):
        strategy = MagicMock()
        cancel = MagicMock(return_value=True)
        adapter = Win32HookAdapter(strategy, cancel=cancel)

        assert adapter._handle_key(True, keys.VK_ESCAPE) is True
        assert adapter._handle_key(False, keys.VK_ESCAPE) is True
        cancel.assert_called_once()
        strategy.on_key_press.assert_not_called()

    def test_escape_reaches_the_app_when_nothing_is_recording(self):
        adapter = Win32HookAdapter(MagicMock(), cancel=lambda: False)

        assert adapter._handle_key(True, keys.VK_ESCAPE) is False
        assert adapter._handle_key(False, keys.VK_ESCAPE) is False

    def test_a_held_escape_cancels_once_and_stays_swallowed_until_released(self):
        cancel = MagicMock(return_value=True)
        adapter = Win32HookAdapter(MagicMock(), cancel=cancel)

        repeats = [adapter._handle_key(True, keys.VK_ESCAPE) for _ in range(30)]
        released = adapter._handle_key(False, keys.VK_ESCAPE)

        assert all(repeats) and released
        cancel.assert_called_once()

    def test_the_next_escape_after_a_cancel_reaches_the_app(self):
        results = iter([True, False])
        adapter = Win32HookAdapter(MagicMock(), cancel=lambda: next(results))
        adapter._handle_key(True, keys.VK_ESCAPE)
        adapter._handle_key(False, keys.VK_ESCAPE)

        assert adapter._handle_key(True, keys.VK_ESCAPE) is False
        assert adapter._handle_key(False, keys.VK_ESCAPE) is False

    def test_other_keys_are_never_swallowed_while_recording(self):
        adapter = Win32HookAdapter(MagicMock(), cancel=lambda: True)

        assert adapter._handle_key(True, 0x4D) is False


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
