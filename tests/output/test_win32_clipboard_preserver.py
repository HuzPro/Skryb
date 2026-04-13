import ctypes
import logging
from unittest.mock import MagicMock, call, patch

import pytest

from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.win32_clipboard_preserver import Win32ClipboardPreserver


_MODULE = "voicetotext.output.win32_clipboard_preserver"
_LOGGER_NAME = _MODULE


@pytest.fixture()
def log_capture(caplog):
    """Attach caplog handler to the module logger (propagate=False in this project)."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.addHandler(caplog.handler)
    caplog.handler.setLevel(logging.DEBUG)
    yield caplog
    logger.removeHandler(caplog.handler)


def _make_preserver(
    user32: MagicMock | None = None,
    kernel32: MagicMock | None = None,
) -> Win32ClipboardPreserver:
    with (
        patch(f"{_MODULE}._user32", user32 or MagicMock()),
        patch(f"{_MODULE}._kernel32", kernel32 or MagicMock()),
    ):
        return Win32ClipboardPreserver()


class TestWin32ClipboardPreserverSave:
    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_open_and_close_clipboard(self, mock_user32, mock_kernel32):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.return_value = 0
        preserver = Win32ClipboardPreserver()

        preserver.save()

        mock_user32.OpenClipboard.assert_called_once_with(None)
        mock_user32.CloseClipboard.assert_called_once()

    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_enumerate_all_clipboard_formats(
        self, mock_user32, mock_kernel32
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.side_effect = [1, 13, 0]
        mock_user32.GetClipboardData.return_value = None
        preserver = Win32ClipboardPreserver()

        preserver.save()

        assert mock_user32.EnumClipboardFormats.call_args_list == [
            call(0),
            call(1),
            call(13),
        ]

    @patch(f"{_MODULE}._string_at", return_value=b"hello")
    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_read_data_for_each_format(
        self, mock_user32, mock_kernel32, mock_string_at
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.side_effect = [1, 0]

        handle = ctypes.c_void_p(12345)
        mock_user32.GetClipboardData.return_value = handle
        mock_kernel32.GlobalSize.return_value = 5
        mock_kernel32.GlobalLock.return_value = ctypes.c_void_p(99999)
        preserver = Win32ClipboardPreserver()

        preserver.save()

        mock_user32.GetClipboardData.assert_called_once_with(1)
        mock_kernel32.GlobalLock.assert_called_once_with(handle)
        mock_kernel32.GlobalSize.assert_called_once_with(handle)
        mock_kernel32.GlobalUnlock.assert_called_once_with(handle)

    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_handle_empty_clipboard(self, mock_user32, mock_kernel32):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.return_value = 0
        preserver = Win32ClipboardPreserver()

        preserver.save()

        mock_user32.GetClipboardData.assert_not_called()

    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_close_clipboard_even_on_error(
        self, mock_user32, mock_kernel32
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.side_effect = OSError("boom")
        preserver = Win32ClipboardPreserver()

        try:
            preserver.save()
        except OSError:
            pass

        mock_user32.CloseClipboard.assert_called_once()

    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_log_warning_when_open_clipboard_fails(
        self, mock_user32, mock_kernel32, log_capture
    ):
        mock_user32.OpenClipboard.return_value = False
        preserver = Win32ClipboardPreserver()

        preserver.save()

        assert "Could not open clipboard" in log_capture.text
        mock_user32.CloseClipboard.assert_not_called()

    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_skip_format_when_get_clipboard_data_returns_null(
        self, mock_user32, mock_kernel32
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.side_effect = [1, 0]
        mock_user32.GetClipboardData.return_value = None
        preserver = Win32ClipboardPreserver()

        preserver.save()

        mock_kernel32.GlobalLock.assert_not_called()


    @patch(f"{_MODULE}._string_at", return_value=b"data")
    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_skip_format_when_global_lock_returns_null(
        self, mock_user32, mock_kernel32, mock_string_at
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.side_effect = [1, 0]
        mock_user32.GetClipboardData.return_value = ctypes.c_void_p(100)
        mock_kernel32.GlobalSize.return_value = 4
        mock_kernel32.GlobalLock.return_value = None
        preserver = Win32ClipboardPreserver()

        preserver.save()

        mock_string_at.assert_not_called()

    @patch(f"{_MODULE}._memmove")
    @patch(f"{_MODULE}._string_at")
    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_overwrite_snapshot_on_second_save(
        self, mock_user32, mock_kernel32, mock_string_at, mock_memmove
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_kernel32.GlobalLock.return_value = ctypes.c_void_p(200)

        # First save: one format
        mock_user32.EnumClipboardFormats.side_effect = [1, 0]
        mock_user32.GetClipboardData.return_value = ctypes.c_void_p(100)
        mock_kernel32.GlobalSize.return_value = 5
        mock_string_at.return_value = b"first"
        preserver = Win32ClipboardPreserver()
        preserver.save()

        # Second save: different format
        mock_user32.EnumClipboardFormats.side_effect = [42, 0]
        mock_string_at.return_value = b"second"
        preserver.save()

        # Restore should only contain the second save's data
        mock_user32.reset_mock()
        mock_kernel32.reset_mock()
        mock_user32.OpenClipboard.return_value = True
        alloc_handle = MagicMock()
        mock_kernel32.GlobalAlloc.return_value = alloc_handle
        mock_kernel32.GlobalLock.return_value = ctypes.c_void_p(600)

        preserver.restore()

        assert mock_user32.SetClipboardData.call_count == 1
        mock_user32.SetClipboardData.assert_called_once_with(42, alloc_handle)


class TestWin32ClipboardPreserverRestore:
    @patch(f"{_MODULE}._memmove")
    @patch(f"{_MODULE}._string_at")
    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_restore_exact_format_ids_and_data_bytes(
        self, mock_user32, mock_kernel32, mock_string_at, mock_memmove
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.side_effect = [1, 13, 0]

        handle = ctypes.c_void_p(100)
        mock_user32.GetClipboardData.return_value = handle
        mock_kernel32.GlobalSize.side_effect = [5, 5]
        mock_kernel32.GlobalLock.return_value = ctypes.c_void_p(200)
        mock_string_at.side_effect = [b"alpha", b"bravo"]

        preserver = Win32ClipboardPreserver()
        preserver.save()

        mock_user32.reset_mock()
        mock_kernel32.reset_mock()
        mock_memmove.reset_mock()
        mock_user32.OpenClipboard.return_value = True
        alloc_handle = ctypes.c_void_p(500)
        mock_kernel32.GlobalAlloc.return_value = alloc_handle
        mock_kernel32.GlobalLock.return_value = ctypes.c_void_p(600)

        preserver.restore()

        assert mock_kernel32.GlobalAlloc.call_args_list == [
            call(0x0002, 5),
            call(0x0002, 5),
        ]
        assert mock_memmove.call_args_list[0][0][1] == b"alpha"
        assert mock_memmove.call_args_list[1][0][1] == b"bravo"
        assert mock_user32.SetClipboardData.call_args_list == [
            call(1, alloc_handle),
            call(13, alloc_handle),
        ]

    @patch(f"{_MODULE}._memmove")
    @patch(f"{_MODULE}._string_at", return_value=b"data")
    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_close_clipboard_even_when_restore_fails(
        self, mock_user32, mock_kernel32, mock_string_at, mock_memmove
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.side_effect = [1, 0]
        handle = ctypes.c_void_p(100)
        mock_user32.GetClipboardData.return_value = handle
        mock_kernel32.GlobalSize.return_value = 4
        mock_kernel32.GlobalLock.return_value = ctypes.c_void_p(200)

        preserver = Win32ClipboardPreserver()
        preserver.save()

        mock_user32.reset_mock()
        mock_kernel32.reset_mock()
        mock_user32.OpenClipboard.return_value = True
        mock_kernel32.GlobalAlloc.side_effect = OSError("alloc failed")

        try:
            preserver.restore()
        except OSError:
            pass

        mock_user32.CloseClipboard.assert_called_once()

    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_noop_when_nothing_was_saved(
        self, mock_user32, mock_kernel32
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.return_value = 0
        preserver = Win32ClipboardPreserver()
        preserver.save()

        mock_user32.reset_mock()

        preserver.restore()

        mock_user32.OpenClipboard.assert_not_called()

    @patch(f"{_MODULE}._string_at", return_value=b"data")
    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_log_warning_when_open_clipboard_fails_on_restore(
        self, mock_user32, mock_kernel32, mock_string_at, log_capture
    ):
        mock_user32.OpenClipboard.return_value = True
        mock_user32.EnumClipboardFormats.side_effect = [1, 0]
        handle = ctypes.c_void_p(100)
        mock_user32.GetClipboardData.return_value = handle
        mock_kernel32.GlobalSize.return_value = 4
        mock_kernel32.GlobalLock.return_value = ctypes.c_void_p(200)

        preserver = Win32ClipboardPreserver()
        preserver.save()

        mock_user32.reset_mock()
        mock_user32.OpenClipboard.return_value = False

        preserver.restore()

        assert "Could not open clipboard" in log_capture.text


    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_noop_when_restore_called_without_save(
        self, mock_user32, mock_kernel32
    ):
        preserver = Win32ClipboardPreserver()

        preserver.restore()

        mock_user32.OpenClipboard.assert_not_called()


class TestWin32ClipboardPreserverIsAClipboardPreserver:
    @patch(f"{_MODULE}._kernel32")
    @patch(f"{_MODULE}._user32")
    def test_should_implement_clipboard_preserver_interface(
        self, mock_user32, mock_kernel32
    ):
        preserver = Win32ClipboardPreserver()

        assert isinstance(preserver, ClipboardPreserver)
