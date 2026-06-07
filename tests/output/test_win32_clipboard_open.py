import pytest

from voicetotext.output import win32_clipboard_open
from voicetotext.output.win32_clipboard_open import open_clipboard_with_retry


class _FakeUser32:
    """Fails ``OpenClipboard`` the first *fail_first* times, then succeeds."""

    def __init__(self, fail_first: int) -> None:
        self._fail_first = fail_first
        self.calls = 0

    def OpenClipboard(self, hwnd: object) -> int:  # noqa: N802 (Win32 name)
        self.calls += 1
        return 0 if self.calls <= self._fail_first else 1


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the bounded retry instant under test.
    monkeypatch.setattr(win32_clipboard_open, "_RETRY_SECONDS", 0.0)


class TestOpenClipboardWithRetry:
    def test_should_succeed_without_retry_when_clipboard_is_free(self) -> None:
        fake = _FakeUser32(fail_first=0)

        assert open_clipboard_with_retry(fake) is True
        assert fake.calls == 1

    def test_should_retry_then_succeed_under_brief_contention(self) -> None:
        fake = _FakeUser32(fail_first=2)

        assert open_clipboard_with_retry(fake) is True
        assert fake.calls == 3

    def test_should_give_up_after_max_attempts(self) -> None:
        fake = _FakeUser32(fail_first=99)

        assert open_clipboard_with_retry(fake) is False
        assert fake.calls == win32_clipboard_open._OPEN_ATTEMPTS
