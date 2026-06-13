"""Tests for :class:`Win32ClipboardWriter`.

Same fake-Win32 strategy as ``test_win32_clipboard_preserver.py``:
an in-memory dict represents the clipboard, method calls are intercepted
via monkeypatch.  We test observable outcomes (what ends up in the
clipboard, what exception is raised) rather than call sequences.
"""

import pytest

from tests.fakes import InstantSleeper
from voicetotext.output.clipboard_write_error import ClipboardWriteError
from voicetotext.output.clipboard_writer import ClipboardWriter
from voicetotext.output.win32_clipboard_writer import Win32ClipboardWriter


_MODULE = "voicetotext.output.win32_clipboard_writer"

_CF_UNICODETEXT = 13


# ---------------------------------------------------------------------------
# Fake Win32 clipboard (write-side only)
# ---------------------------------------------------------------------------


class _FakeWin32Clipboard:
    """Minimal fake covering only the operations Win32ClipboardWriter uses."""

    def __init__(self) -> None:
        self.open_should_fail: bool = False
        self.global_alloc_should_fail: bool = False
        self.global_lock_should_fail: bool = False
        self.set_clipboard_data_should_fail: bool = False
        self._next_handle: int = 1
        self._memory: dict[int, bytes] = {}
        self.clipboard: dict[int, bytes] = {}

    # --- user32 ---

    def OpenClipboard(self, hwnd) -> int:  # noqa: N802
        return 0 if self.open_should_fail else 1

    def CloseClipboard(self) -> int:  # noqa: N802
        return 1

    def EmptyClipboard(self) -> int:  # noqa: N802
        self.clipboard.clear()
        return 1

    def SetClipboardData(self, fmt: int, handle):  # noqa: N802
        if self.set_clipboard_data_should_fail:
            return None
        self.clipboard[fmt] = self._memory.get(handle, b"")
        return handle

    # --- kernel32 ---

    def GlobalAlloc(self, flags: int, size: int):  # noqa: N802
        if self.global_alloc_should_fail:
            return 0
        handle = self._next_handle
        self._next_handle += 1
        self._memory[handle] = b"\x00" * size
        return handle

    def GlobalLock(self, handle):  # noqa: N802
        if self.global_lock_should_fail:
            return 0
        return handle

    def GlobalUnlock(self, handle) -> int:  # noqa: N802
        return 1

    # --- ctypes helper ---

    def memmove(self, dest, src, size: int) -> None:
        if isinstance(src, (bytes, bytearray)):
            self._memory[dest] = bytes(src)[:size]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_clipboard(monkeypatch: pytest.MonkeyPatch) -> _FakeWin32Clipboard:
    from types import SimpleNamespace

    fake = _FakeWin32Clipboard()
    fake_user32 = SimpleNamespace(
        OpenClipboard=fake.OpenClipboard,
        CloseClipboard=fake.CloseClipboard,
        EmptyClipboard=fake.EmptyClipboard,
        SetClipboardData=fake.SetClipboardData,
    )
    fake_kernel32 = SimpleNamespace(
        GlobalAlloc=fake.GlobalAlloc,
        GlobalLock=fake.GlobalLock,
        GlobalUnlock=fake.GlobalUnlock,
    )
    monkeypatch.setattr(f"{_MODULE}._user32", fake_user32)
    monkeypatch.setattr(f"{_MODULE}._kernel32", fake_kernel32)
    monkeypatch.setattr(f"{_MODULE}._memmove", fake.memmove)
    return fake


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


class TestWin32ClipboardWriterSetsText:
    def test_should_write_text_to_clipboard_as_unicodetext(
        self, fake_clipboard: _FakeWin32Clipboard
    ) -> None:
        writer = Win32ClipboardWriter(InstantSleeper())

        writer.set_text("hello")

        assert _CF_UNICODETEXT in fake_clipboard.clipboard
        assert fake_clipboard.clipboard[_CF_UNICODETEXT] == "hello\0".encode("utf-16-le")

    def test_should_write_empty_string_without_crashing(
        self, fake_clipboard: _FakeWin32Clipboard
    ) -> None:
        writer = Win32ClipboardWriter(InstantSleeper())

        writer.set_text("")

        assert _CF_UNICODETEXT in fake_clipboard.clipboard

    def test_should_write_unicode_text_correctly(
        self, fake_clipboard: _FakeWin32Clipboard
    ) -> None:
        writer = Win32ClipboardWriter(InstantSleeper())

        writer.set_text("café ✓")

        assert fake_clipboard.clipboard[_CF_UNICODETEXT] == "café ✓\0".encode("utf-16-le")


class TestWin32ClipboardWriterFailures:
    def test_should_raise_when_open_clipboard_fails(
        self, fake_clipboard: _FakeWin32Clipboard
    ) -> None:
        fake_clipboard.open_should_fail = True
        writer = Win32ClipboardWriter(InstantSleeper())

        with pytest.raises(ClipboardWriteError):
            writer.set_text("text")

    def test_should_raise_when_global_alloc_fails(
        self, fake_clipboard: _FakeWin32Clipboard
    ) -> None:
        fake_clipboard.global_alloc_should_fail = True
        writer = Win32ClipboardWriter(InstantSleeper())

        with pytest.raises(ClipboardWriteError):
            writer.set_text("text")

    def test_should_raise_when_global_lock_fails(
        self, fake_clipboard: _FakeWin32Clipboard
    ) -> None:
        # This is the root cause of the production bug: GlobalLock can return
        # NULL with errno == 0, which pyperclip's CheckedCall does not catch,
        # leading to memmove(NULL, ...) and a silent process-killing segfault.
        # Win32ClipboardWriter must raise ClipboardWriteError instead.
        fake_clipboard.global_lock_should_fail = True
        writer = Win32ClipboardWriter(InstantSleeper())

        with pytest.raises(ClipboardWriteError):
            writer.set_text("text")

    def test_should_raise_when_set_clipboard_data_fails(
        self, fake_clipboard: _FakeWin32Clipboard
    ) -> None:
        fake_clipboard.set_clipboard_data_should_fail = True
        writer = Win32ClipboardWriter(InstantSleeper())

        with pytest.raises(ClipboardWriteError):
            writer.set_text("text")


class TestWin32ClipboardWriterIsAClipboardWriter:
    def test_should_satisfy_clipboard_writer_protocol(self) -> None:
        writer = Win32ClipboardWriter(InstantSleeper())

        assert isinstance(writer, ClipboardWriter)
