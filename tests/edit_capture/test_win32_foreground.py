from unittest.mock import patch

from voicetotext.edit_capture import win32_foreground as wf
from voicetotext.edit_capture.win32_foreground import (
    UNKNOWN_WINDOW,
    ForegroundWindow,
    foreground_window,
)


_HWND = 0x1234
_PID = 4321
_HANDLE = 99


def _write_pid(pid: int):
    def _call(hwnd, pid_ref):
        pid_ref._obj.value = pid
        return 7  # thread id, unused here

    return _call


def _write_image_name(path: str):
    def _call(handle, flags, buffer, size_ref):
        buffer.value = path
        return 1

    return _call


def _write_title(title: str):
    def _call(hwnd, buffer, length):
        buffer.value = title
        return len(title)

    return _call


def _wire_a_real_window(
    mock_user32,
    mock_kernel32,
    image: str = r"C:\Program Files\Claude\Claude.exe",
    title: str = "Claude",
) -> None:
    mock_user32.GetForegroundWindow.return_value = _HWND
    mock_user32.GetWindowThreadProcessId.side_effect = _write_pid(_PID)
    mock_user32.GetWindowTextW.side_effect = _write_title(title)
    mock_kernel32.OpenProcess.return_value = _HANDLE
    mock_kernel32.QueryFullProcessImageNameW.side_effect = _write_image_name(image)


@patch.object(wf, "_kernel32")
@patch.object(wf, "_user32")
class TestForegroundWindowDescribesTheFocusedWindow:
    def test_should_report_hwnd_pid_process_and_title(
        self, mock_user32, mock_kernel32,
    ):
        _wire_a_real_window(mock_user32, mock_kernel32)

        assert foreground_window() == ForegroundWindow(
            hwnd=_HWND, pid=_PID, process="claude.exe", title="Claude",
        )

    def test_should_reduce_the_image_path_to_a_lowercased_base_name(
        self, mock_user32, mock_kernel32,
    ):
        _wire_a_real_window(
            mock_user32, mock_kernel32, image=r"D:\Apps\Sub Dir\NotePad.EXE",
        )

        assert foreground_window().process == "notepad.exe"

    def test_should_open_the_process_for_limited_query_only(
        self, mock_user32, mock_kernel32,
    ):
        # PROCESS_QUERY_INFORMATION is denied for elevated targets; the
        # LIMITED right is granted to any process the user can see.
        _wire_a_real_window(mock_user32, mock_kernel32)

        foreground_window()

        access, _, pid = mock_kernel32.OpenProcess.call_args.args
        assert access == wf._PROCESS_QUERY_LIMITED_INFORMATION
        assert pid == _PID


@patch.object(wf, "_kernel32")
@patch.object(wf, "_user32")
class TestForegroundWindowTreatsFailureAsUnknown:
    def test_should_report_unknown_when_no_window_has_focus(
        self, mock_user32, mock_kernel32,
    ):
        _wire_a_real_window(mock_user32, mock_kernel32)
        mock_user32.GetForegroundWindow.return_value = 0

        assert foreground_window() == UNKNOWN_WINDOW

    def test_should_report_unknown_when_the_window_has_no_process(
        self, mock_user32, mock_kernel32,
    ):
        _wire_a_real_window(mock_user32, mock_kernel32)
        mock_user32.GetWindowThreadProcessId.side_effect = _write_pid(0)

        assert foreground_window() == UNKNOWN_WINDOW

    def test_should_report_unknown_when_the_process_cannot_be_opened(
        self, mock_user32, mock_kernel32,
    ):
        _wire_a_real_window(mock_user32, mock_kernel32)
        mock_kernel32.OpenProcess.return_value = 0

        assert foreground_window() == UNKNOWN_WINDOW

    def test_should_report_unknown_when_the_image_name_query_fails(
        self, mock_user32, mock_kernel32,
    ):
        _wire_a_real_window(mock_user32, mock_kernel32)
        mock_kernel32.QueryFullProcessImageNameW.side_effect = None
        mock_kernel32.QueryFullProcessImageNameW.return_value = 0

        assert foreground_window() == UNKNOWN_WINDOW

    def test_should_report_unknown_instead_of_raising(
        self, mock_user32, mock_kernel32,
    ):
        mock_user32.GetForegroundWindow.side_effect = OSError("access denied")

        assert foreground_window() == UNKNOWN_WINDOW


@patch.object(wf, "_kernel32")
@patch.object(wf, "_user32")
class TestForegroundWindowReleasesTheProcessHandle:
    def test_should_close_the_handle_after_a_successful_query(
        self, mock_user32, mock_kernel32,
    ):
        _wire_a_real_window(mock_user32, mock_kernel32)

        foreground_window()

        mock_kernel32.CloseHandle.assert_called_once_with(_HANDLE)

    def test_should_close_the_handle_when_the_query_fails(
        self, mock_user32, mock_kernel32,
    ):
        _wire_a_real_window(mock_user32, mock_kernel32)
        mock_kernel32.QueryFullProcessImageNameW.side_effect = OSError("boom")

        foreground_window()

        mock_kernel32.CloseHandle.assert_called_once_with(_HANDLE)

    def test_should_not_close_a_handle_it_never_opened(
        self, mock_user32, mock_kernel32,
    ):
        _wire_a_real_window(mock_user32, mock_kernel32)
        mock_kernel32.OpenProcess.return_value = 0

        foreground_window()

        mock_kernel32.CloseHandle.assert_not_called()
