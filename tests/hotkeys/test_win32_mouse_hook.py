import ctypes
from unittest.mock import MagicMock, patch

import pytest

from voicetotext.hotkeys import keys
from voicetotext.hotkeys import win32_keyboard_hook as wkh
from voicetotext.hotkeys import win32_mouse_hook as wmh
from voicetotext.hotkeys.win32_mouse_hook import Win32MouseHook

_WM_MOUSEMOVE = 0x0200
_WM_LBUTTONDOWN = 0x0201
_WM_RBUTTONUP = 0x0205
_WM_MOUSEWHEEL = 0x020A


def _lparam(mouse_data: int = 0):
    info = wmh._MSLLHOOKSTRUCT(mouseData=mouse_data)
    return info, ctypes.addressof(info)


@pytest.mark.parametrize(("message", "mouse_data", "expected"), [
    (wmh._WM_MBUTTONDOWN, 0, (True, keys.VK_MBUTTON)),
    (wmh._WM_MBUTTONUP, 0, (False, keys.VK_MBUTTON)),
    (wmh._WM_XBUTTONDOWN, 0x0001 << 16, (True, keys.VK_XBUTTON1)),
    (wmh._WM_XBUTTONUP, 0x0001 << 16, (False, keys.VK_XBUTTON1)),
    (wmh._WM_XBUTTONDOWN, 0x0002 << 16, (True, keys.VK_XBUTTON2)),
    (wmh._WM_XBUTTONUP, 0x0002 << 16, (False, keys.VK_XBUTTON2)),
])
def test_buttons_three_to_five_reach_the_handler_as_keys(message, mouse_data, expected):
    handler = MagicMock(return_value=False)
    info, lparam = _lparam(mouse_data)

    Win32MouseHook(handler)._dispatch(message, lparam)

    handler.assert_called_once_with(*expected)


def test_moves_wheel_and_left_right_clicks_never_reach_the_handler():
    handler = MagicMock(return_value=True)
    hook = Win32MouseHook(handler)
    info, lparam = _lparam()

    swallowed = [
        hook._dispatch(message, lparam)
        for message in [_WM_MOUSEMOVE] * 2000 + [_WM_LBUTTONDOWN, _WM_RBUTTONUP, _WM_MOUSEWHEEL]
    ]

    assert not any(swallowed)
    handler.assert_not_called()


@patch.object(wkh, "_user32")
def test_a_button_the_handler_claims_is_swallowed(mock_user32):
    hook = Win32MouseHook(MagicMock(return_value=True))
    info, lparam = _lparam(0x0001 << 16)

    result = hook._hook_proc(wkh._HC_ACTION, wmh._WM_XBUTTONDOWN, lparam)

    assert result == wkh._SUPPRESS
    mock_user32.CallNextHookEx.assert_not_called()


@patch.object(wkh, "_user32")
def test_a_button_the_handler_leaves_is_passed_on(mock_user32):
    mock_user32.CallNextHookEx.return_value = 0
    hook = Win32MouseHook(MagicMock(return_value=False))
    info, lparam = _lparam(0x0002 << 16)

    hook._hook_proc(wkh._HC_ACTION, wmh._WM_XBUTTONUP, lparam)

    mock_user32.CallNextHookEx.assert_called_once()


@patch.object(wkh, "_kernel32")
@patch.object(wkh, "_user32")
def test_start_installs_a_mouse_hook_and_stop_removes_it(mock_user32, mock_kernel32):
    mock_user32.SetWindowsHookExW.return_value = 777
    mock_user32.GetMessageW.return_value = 0
    mock_kernel32.GetCurrentThreadId.return_value = 42
    hook = Win32MouseHook(MagicMock())

    hook.start()
    hook.stop()

    assert mock_user32.SetWindowsHookExW.call_args.args[0] == wmh._WH_MOUSE_LL
    mock_user32.UnhookWindowsHookEx.assert_called_once_with(777)
