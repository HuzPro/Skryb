import ctypes
from unittest.mock import MagicMock, patch

import pytest

from voicetotext.core.exceptions import HotkeyError
from voicetotext.hotkeys import win32_keyboard_hook as wkh
from voicetotext.hotkeys.win32_keyboard_hook import Win32KeyboardHook


def _lparam(vk: int, flags: int = 0, extra: int = 0):
    # Returned struct must stay referenced by the caller so its address
    # remains valid for the duration of the call under test.
    info = wkh._KBDLLHOOKSTRUCT(vkCode=vk, flags=flags, dwExtraInfo=extra)
    return info, ctypes.addressof(info)


class TestWin32KeyboardHookDispatch:
    def test_should_report_key_down_to_handler(self):
        handler = MagicMock(return_value=False)
        hook = Win32KeyboardHook(handler)
        info, lparam = _lparam(0x4D)

        hook._dispatch(wkh._WM_KEYDOWN, lparam)

        handler.assert_called_once_with(True, 0x4D)

    def test_should_report_sys_key_down_to_handler(self):
        # Alt chords arrive as WM_SYSKEYDOWN; missing this drops them.
        handler = MagicMock(return_value=False)
        hook = Win32KeyboardHook(handler)
        info, lparam = _lparam(0xA4)

        hook._dispatch(wkh._WM_SYSKEYDOWN, lparam)

        handler.assert_called_once_with(True, 0xA4)

    def test_should_report_key_up_to_handler(self):
        handler = MagicMock(return_value=False)
        hook = Win32KeyboardHook(handler)
        info, lparam = _lparam(0x4D)

        hook._dispatch(wkh._WM_KEYUP, lparam)

        handler.assert_called_once_with(False, 0x4D)

    def test_should_report_sys_key_up_to_handler(self):
        handler = MagicMock(return_value=False)
        hook = Win32KeyboardHook(handler)
        info, lparam = _lparam(0xA4)

        hook._dispatch(wkh._WM_SYSKEYUP, lparam)

        handler.assert_called_once_with(False, 0xA4)


class TestWin32KeyboardHookObserver:
    def test_should_report_key_down_with_flags_and_extra_info(self):
        observer = MagicMock()
        hook = Win32KeyboardHook(MagicMock(return_value=False), observer)
        info, lparam = _lparam(0x4D, flags=0x10, extra=0x56325458)

        hook._dispatch(wkh._WM_KEYDOWN, lparam)

        observer.assert_called_once_with(True, 0x4D, 0x10, 0x56325458)

    def test_should_report_key_up_with_flags_and_extra_info(self):
        observer = MagicMock()
        hook = Win32KeyboardHook(MagicMock(return_value=False), observer)
        info, lparam = _lparam(0x4D, flags=0x90, extra=7)

        hook._dispatch(wkh._WM_KEYUP, lparam)

        observer.assert_called_once_with(False, 0x4D, 0x90, 7)

    def test_should_observe_after_the_handler_has_decided(self):
        calls = []
        handler = MagicMock(side_effect=lambda *_: calls.append("handler") or False)
        observer = MagicMock(side_effect=lambda *_: calls.append("observer"))
        hook = Win32KeyboardHook(handler, observer)
        info, lparam = _lparam(0x4D)

        hook._dispatch(wkh._WM_KEYDOWN, lparam)

        assert calls == ["handler", "observer"]

    def test_should_let_the_key_through_when_the_observer_raises(self):
        observer = MagicMock(side_effect=RuntimeError("observer is broken"))
        hook = Win32KeyboardHook(MagicMock(return_value=False), observer)
        info, lparam = _lparam(0x4D)

        assert hook._dispatch(wkh._WM_KEYDOWN, lparam) is False

    def test_should_not_report_a_key_it_swallowed(self):
        observer = MagicMock()
        hook = Win32KeyboardHook(MagicMock(return_value=True), observer)
        info, lparam = _lparam(0x1B)

        assert hook._dispatch(wkh._WM_KEYDOWN, lparam) is True
        observer.assert_not_called()

    def test_should_not_observe_keys_when_no_observer_was_given(self):
        handler = MagicMock(return_value=False)
        hook = Win32KeyboardHook(handler)
        info, lparam = _lparam(0x4D)

        hook._dispatch(wkh._WM_KEYDOWN, lparam)

        handler.assert_called_once_with(True, 0x4D)


class TestWin32KeyboardHookProc:
    @patch.object(wkh, "_user32")
    def test_should_suppress_when_handler_returns_true(self, mock_user32):
        hook = Win32KeyboardHook(MagicMock(return_value=True))
        info = wkh._KBDLLHOOKSTRUCT(vkCode=0x4D)

        result = hook._hook_proc(
            wkh._HC_ACTION, wkh._WM_KEYDOWN, ctypes.addressof(info),
        )

        assert result == wkh._SUPPRESS
        mock_user32.CallNextHookEx.assert_not_called()

    @patch.object(wkh, "_user32")
    def test_should_chain_to_next_hook_when_handler_returns_false(self, mock_user32):
        mock_user32.CallNextHookEx.return_value = 0
        hook = Win32KeyboardHook(MagicMock(return_value=False))
        info = wkh._KBDLLHOOKSTRUCT(vkCode=0x4D)

        hook._hook_proc(wkh._HC_ACTION, wkh._WM_KEYDOWN, ctypes.addressof(info))

        mock_user32.CallNextHookEx.assert_called_once()

    @patch.object(wkh, "_user32")
    def test_should_chain_without_dispatch_for_non_action_code(self, mock_user32):
        handler = MagicMock()
        hook = Win32KeyboardHook(handler)

        hook._hook_proc(3, wkh._WM_KEYDOWN, 0)

        handler.assert_not_called()
        mock_user32.CallNextHookEx.assert_called_once()


class TestWin32KeyboardHookLifecycle:
    @patch.object(wkh, "_kernel32")
    @patch.object(wkh, "_user32")
    def test_should_install_then_uninstall_hook_across_start_stop(
        self, mock_user32, mock_kernel32,
    ):
        mock_user32.SetWindowsHookExW.return_value = 4321
        mock_user32.GetMessageW.return_value = 0  # pump exits immediately
        mock_kernel32.GetCurrentThreadId.return_value = 99
        hook = Win32KeyboardHook(MagicMock())

        hook.start()
        hook.stop()

        mock_user32.SetWindowsHookExW.assert_called_once()
        mock_user32.UnhookWindowsHookEx.assert_called_once_with(4321)
        assert hook._thread is None

    @patch.object(wkh, "_kernel32")
    @patch.object(wkh, "_user32")
    def test_should_install_only_once_when_started_twice(
        self, mock_user32, mock_kernel32,
    ):
        mock_user32.SetWindowsHookExW.return_value = 4321
        mock_user32.GetMessageW.return_value = 0
        mock_kernel32.GetCurrentThreadId.return_value = 99
        hook = Win32KeyboardHook(MagicMock())

        hook.start()
        hook.start()
        hook.stop()

        assert mock_user32.SetWindowsHookExW.call_count == 1

    @patch.object(wkh, "_user32")
    def test_should_be_safe_to_stop_when_never_started(self, mock_user32):
        hook = Win32KeyboardHook(MagicMock())

        hook.stop()

        mock_user32.PostThreadMessageW.assert_not_called()

    @patch.object(wkh, "_kernel32")
    @patch.object(wkh, "_user32")
    def test_should_raise_and_not_pump_when_hook_install_fails(
        self, mock_user32, mock_kernel32,
    ):
        mock_user32.SetWindowsHookExW.return_value = 0  # NULL handle == failure
        mock_kernel32.GetCurrentThreadId.return_value = 99
        hook = Win32KeyboardHook(MagicMock())

        with pytest.raises(HotkeyError):
            hook.start()

        assert hook._thread is None
        mock_user32.GetMessageW.assert_not_called()
