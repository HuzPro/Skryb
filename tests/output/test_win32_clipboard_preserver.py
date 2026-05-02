"""Tests for :class:`Win32ClipboardPreserver`.

The test strategy is behavioural: we stand up a fake Win32 clipboard
(an in-memory ``dict[int, bytes]`` mapping format id → data) and
verify save/restore round-trips.  We do *not* assert on exact ctypes
call orders — that's implementation detail.

One dedicated section guards the ctypes signature declarations.
Those are a real integration concern: on 64-bit Python, missing
``restype`` annotations silently truncate HANDLE values to 4 bytes,
which crashes ``save()`` non-obviously at runtime.
"""

import ctypes
import ctypes.wintypes as _wt
import logging
from types import SimpleNamespace

import pytest

from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.win32_clipboard_preserver import Win32ClipboardPreserver


_MODULE = "voicetotext.output.win32_clipboard_preserver"


# ---------------------------------------------------------------------------
# Fake Win32 clipboard
# ---------------------------------------------------------------------------


class _FakeWin32Clipboard:
    """In-memory stand-in for the Windows clipboard.

    Mimics the public behaviour of the subset of ``user32`` and
    ``kernel32`` that :class:`Win32ClipboardPreserver` uses.  Data is
    stored as ``format_id -> bytes``.  ``GlobalLock`` returns the
    handle itself as the "pointer" — since the fake's
    ``string_at``/``memmove`` replacements look up the handle
    directly, there is no real pointer arithmetic to model.
    """

    def __init__(
        self, initial: dict[int, bytes] | None = None,
    ) -> None:
        self.contents: dict[int, bytes] = dict(initial or {})
        self.open_clipboard_should_succeed: bool = True
        self.fail_global_lock_calls: int = 0
        self._handles: dict[int, bytes] = {}
        self._next_handle: int = 1

    # --- fake user32 -------------------------------------------------

    def OpenClipboard(self, hwnd) -> int:  # noqa: N802 (Win32 name)
        return 1 if self.open_clipboard_should_succeed else 0

    def CloseClipboard(self) -> int:  # noqa: N802
        return 1

    def EmptyClipboard(self) -> int:  # noqa: N802
        self.contents.clear()
        return 1

    def EnumClipboardFormats(self, prev: int) -> int:  # noqa: N802
        formats = sorted(self.contents.keys())
        if prev == 0:
            return formats[0] if formats else 0
        try:
            idx = formats.index(prev) + 1
        except ValueError:
            return 0
        return formats[idx] if idx < len(formats) else 0

    def GetClipboardData(self, fmt: int):  # noqa: N802
        if fmt not in self.contents:
            return None
        handle = self._next_handle
        self._next_handle += 1
        self._handles[handle] = self.contents[fmt]
        return handle

    def SetClipboardData(self, fmt: int, handle):  # noqa: N802
        self.contents[fmt] = self._handles[handle]
        return handle

    # --- fake kernel32 -----------------------------------------------

    def GlobalSize(self, handle) -> int:  # noqa: N802
        return len(self._handles[handle])

    def GlobalLock(self, handle):  # noqa: N802
        if self.fail_global_lock_calls > 0:
            self.fail_global_lock_calls -= 1
            return 0
        return handle

    def GlobalUnlock(self, handle) -> int:  # noqa: N802
        return 1

    def GlobalAlloc(self, flags: int, size: int):  # noqa: N802
        handle = self._next_handle
        self._next_handle += 1
        self._handles[handle] = b"\x00" * size
        return handle

    # --- fake ctypes helpers -----------------------------------------

    def string_at(self, ptr, size: int) -> bytes:
        return self._handles[ptr][:size]

    def memmove(self, dest, src, size: int) -> None:
        self._handles[dest] = bytes(src)[:size]


@pytest.fixture()
def fake_clipboard(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeWin32Clipboard:
    fake = _FakeWin32Clipboard()
    fake_user32 = SimpleNamespace(
        OpenClipboard=fake.OpenClipboard,
        CloseClipboard=fake.CloseClipboard,
        EmptyClipboard=fake.EmptyClipboard,
        EnumClipboardFormats=fake.EnumClipboardFormats,
        GetClipboardData=fake.GetClipboardData,
        SetClipboardData=fake.SetClipboardData,
    )
    fake_kernel32 = SimpleNamespace(
        GlobalSize=fake.GlobalSize,
        GlobalLock=fake.GlobalLock,
        GlobalUnlock=fake.GlobalUnlock,
        GlobalAlloc=fake.GlobalAlloc,
    )
    monkeypatch.setattr(f"{_MODULE}._user32", fake_user32)
    monkeypatch.setattr(f"{_MODULE}._kernel32", fake_kernel32)
    monkeypatch.setattr(f"{_MODULE}._string_at", fake.string_at)
    monkeypatch.setattr(f"{_MODULE}._memmove", fake.memmove)
    return fake


@pytest.fixture()
def log_capture(caplog: pytest.LogCaptureFixture):
    logger = logging.getLogger(_MODULE)
    logger.addHandler(caplog.handler)
    caplog.handler.setLevel(logging.DEBUG)
    yield caplog
    logger.removeHandler(caplog.handler)


# ---------------------------------------------------------------------------
# Behavioural tests — save/restore round-trip
# ---------------------------------------------------------------------------


class TestSaveAndRestoreRoundTrip:
    def test_should_restore_single_format_clipboard(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        fake_clipboard.contents = {1: b"hello world"}
        preserver = Win32ClipboardPreserver()

        preserver.save()
        fake_clipboard.contents = {1: b"transcription text"}
        preserver.restore()

        assert fake_clipboard.contents == {1: b"hello world"}

    def test_should_restore_multi_format_clipboard(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        fake_clipboard.contents = {
            1: b"plain text",
            13: b"unicode text encoded",
            49161: b"<html>rich</html>",
        }
        preserver = Win32ClipboardPreserver()

        preserver.save()
        fake_clipboard.contents = {1: b"garbage"}
        preserver.restore()

        assert fake_clipboard.contents == {
            1: b"plain text",
            13: b"unicode text encoded",
            49161: b"<html>rich</html>",
        }

    def test_second_save_overwrites_first(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        preserver = Win32ClipboardPreserver()

        fake_clipboard.contents = {1: b"first"}
        preserver.save()
        fake_clipboard.contents = {0xC042: b"second"}
        preserver.save()

        fake_clipboard.contents = {0xC099: b"cleared"}
        preserver.restore()

        assert fake_clipboard.contents == {0xC042: b"second"}


class TestSaveEdgeCases:
    def test_should_handle_empty_clipboard(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        fake_clipboard.contents = {}
        preserver = Win32ClipboardPreserver()

        preserver.save()

        # A subsequent restore is a no-op and must not crash.
        preserver.restore()

    def test_should_skip_cf_bitmap_to_avoid_segfault_on_hbitmap(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        # CF_BITMAP (2) returns an HBITMAP, NOT an HGLOBAL, so calling
        # GlobalSize/GlobalLock on it is undefined behaviour — at best
        # returns garbage, at worst segfaults the process during
        # _string_at(ptr, garbage_size). Skip it entirely.
        fake_clipboard.contents = {
            2: b"\x00" * 16,  # CF_BITMAP — must be skipped
            13: b"safe text",  # CF_UNICODETEXT — must still be preserved
        }
        preserver = Win32ClipboardPreserver()

        preserver.save()
        fake_clipboard.contents = {1: b"replaced"}
        preserver.restore()

        assert 2 not in fake_clipboard.contents
        assert fake_clipboard.contents[13] == b"safe text"

    def test_should_skip_cf_enhmetafile_and_other_non_hglobal_formats(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        # CF_METAFILEPICT (3), CF_PALETTE (9), CF_ENHMETAFILE (14),
        # CF_OWNERDISPLAY (0x80), CF_DSPBITMAP (0x82), CF_DSPMETAFILEPICT
        # (0x83), CF_DSPENHMETAFILE (0x8E) are non-HGLOBAL and must
        # never be read with GlobalSize/_string_at.
        fake_clipboard.contents = {
            3: b"x", 9: b"x", 14: b"x",
            0x80: b"x", 0x82: b"x", 0x83: b"x", 0x8E: b"x",
            13: b"safe text",
        }
        preserver = Win32ClipboardPreserver()

        preserver.save()
        fake_clipboard.contents = {}
        preserver.restore()

        assert set(fake_clipboard.contents.keys()) == {13}

    def test_should_preserve_registered_formats_above_c000(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        # Registered formats (>= 0xC000) are HGLOBAL by Win32 convention
        # and are how apps put HTML/RTF/file-list custom data on the
        # clipboard — must continue to round-trip.
        fake_clipboard.contents = {
            0xC001: b"<html>preserved</html>",
            0xC123: b"some-app-format",
            13: b"plain unicode text",
        }
        preserver = Win32ClipboardPreserver()

        preserver.save()
        fake_clipboard.contents = {}
        preserver.restore()

        assert fake_clipboard.contents == {
            0xC001: b"<html>preserved</html>",
            0xC123: b"some-app-format",
            13: b"plain unicode text",
        }

    def test_should_log_warning_when_open_clipboard_fails(
        self, fake_clipboard: _FakeWin32Clipboard, log_capture
    ):
        fake_clipboard.open_clipboard_should_succeed = False
        preserver = Win32ClipboardPreserver()

        preserver.save()

        assert "Could not open clipboard" in log_capture.text


class TestRestoreEdgeCases:
    def test_should_be_a_noop_when_save_was_never_called(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        fake_clipboard.contents = {1: b"untouched"}
        preserver = Win32ClipboardPreserver()

        preserver.restore()

        assert fake_clipboard.contents == {1: b"untouched"}

    def test_should_be_a_noop_when_last_save_was_empty(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        fake_clipboard.contents = {}
        preserver = Win32ClipboardPreserver()
        preserver.save()

        fake_clipboard.contents = {1: b"new"}
        preserver.restore()

        assert fake_clipboard.contents == {1: b"new"}

    def test_should_log_warning_when_open_clipboard_fails_on_restore(
        self, fake_clipboard: _FakeWin32Clipboard, log_capture
    ):
        fake_clipboard.contents = {1: b"saved"}
        preserver = Win32ClipboardPreserver()
        preserver.save()

        fake_clipboard.open_clipboard_should_succeed = False
        preserver.restore()

        assert "Could not open clipboard" in log_capture.text

    def test_should_skip_format_when_global_lock_fails_during_allocation(
        self, fake_clipboard: _FakeWin32Clipboard, log_capture
    ):
        fake_clipboard.contents = {1: b"saved data"}
        preserver = Win32ClipboardPreserver()
        preserver.save()

        fake_clipboard.contents = {}
        fake_clipboard.fail_global_lock_calls = 1
        preserver.restore()

        assert fake_clipboard.contents == {}
        assert "GlobalLock" in log_capture.text


class TestInterfaceConformance:
    def test_should_implement_clipboard_preserver_interface(
        self, fake_clipboard: _FakeWin32Clipboard
    ):
        preserver = Win32ClipboardPreserver()

        assert isinstance(preserver, ClipboardPreserver)


# ---------------------------------------------------------------------------
# ctypes signature guards
# ---------------------------------------------------------------------------
#
# These tests protect against the 64-bit handle truncation bug.  They
# must NOT patch ``_user32``/``_kernel32`` because they read the real
# ctypes function objects to verify their argtypes/restype.


class TestWin32ClipboardPreserverCtypesSignatures:
    @pytest.mark.parametrize(
        ("func_name", "expected_restype"),
        [
            ("OpenClipboard", _wt.BOOL),
            ("CloseClipboard", _wt.BOOL),
            ("EmptyClipboard", _wt.BOOL),
            ("EnumClipboardFormats", _wt.UINT),
            ("GetClipboardData", _wt.HANDLE),
            ("SetClipboardData", _wt.HANDLE),
        ],
    )
    def test_user32_function_has_correct_restype(
        self, func_name: str, expected_restype
    ):
        from voicetotext.output.win32_clipboard_preserver import _user32

        assert getattr(_user32, func_name).restype is expected_restype

    @pytest.mark.parametrize(
        ("func_name", "expected_argtypes"),
        [
            ("OpenClipboard", [_wt.HWND]),
            ("EnumClipboardFormats", [_wt.UINT]),
            ("GetClipboardData", [_wt.UINT]),
        ],
    )
    def test_user32_function_has_correct_argtypes(
        self, func_name: str, expected_argtypes
    ):
        from voicetotext.output.win32_clipboard_preserver import _user32

        assert getattr(_user32, func_name).argtypes == expected_argtypes

    @pytest.mark.parametrize(
        ("func_name", "expected_restype"),
        [
            ("GlobalSize", ctypes.c_size_t),
            ("GlobalLock", ctypes.c_void_p),
            ("GlobalUnlock", _wt.BOOL),
            ("GlobalAlloc", _wt.HGLOBAL),
        ],
    )
    def test_kernel32_function_has_correct_restype(
        self, func_name: str, expected_restype
    ):
        from voicetotext.output.win32_clipboard_preserver import _kernel32

        assert getattr(_kernel32, func_name).restype is expected_restype

    @pytest.mark.parametrize(
        ("func_name", "expected_argtypes"),
        [
            ("GlobalSize", [_wt.HGLOBAL]),
            ("GlobalLock", [_wt.HGLOBAL]),
        ],
    )
    def test_kernel32_function_has_correct_argtypes(
        self, func_name: str, expected_argtypes
    ):
        from voicetotext.output.win32_clipboard_preserver import _kernel32

        assert getattr(_kernel32, func_name).argtypes == expected_argtypes

    def test_kernel32_global_alloc_accepts_size_t_for_byte_count(self):
        from voicetotext.output.win32_clipboard_preserver import _kernel32

        assert _kernel32.GlobalAlloc.argtypes[1] is ctypes.c_size_t
