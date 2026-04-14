from voicetotext.output.clipboard_preserver import ClipboardPreserver
from voicetotext.output.handler import TextOutputHandler
from voicetotext.output.sleeper import Sleeper


class FakeTextOutputHandler(TextOutputHandler):
    """Captures every string passed to :meth:`output` in :attr:`outputs`."""

    def __init__(self) -> None:
        self.outputs: list[str] = []
        self._error: Exception | None = None

    def will_fail(self, error: Exception) -> None:
        self._error = error

    def output(self, text: str) -> None:
        if self._error is not None:
            raise self._error
        self.outputs.append(text)


class FakeClipboardPreserver(ClipboardPreserver):
    """In-memory ClipboardPreserver.

    Tracks save/restore call counts and lets tests simulate failures
    in either operation independently.
    """

    def __init__(self) -> None:
        self.saved_count: int = 0
        self.restored_count: int = 0
        self._save_error: Exception | None = None
        self._restore_error: Exception | None = None

    def will_fail_save(self, error: Exception) -> None:
        self._save_error = error

    def will_fail_restore(self, error: Exception) -> None:
        self._restore_error = error

    def save(self) -> None:
        if self._save_error is not None:
            raise self._save_error
        self.saved_count += 1

    def restore(self) -> None:
        if self._restore_error is not None:
            raise self._restore_error
        self.restored_count += 1


class InstantSleeper(Sleeper):
    """Sleeper that returns immediately but records every requested duration.

    Lets tests assert on the total time that *would have* been slept
    without actually slowing the suite down.
    """

    def __init__(self) -> None:
        self.requested_durations: list[float] = []

    @property
    def total_requested(self) -> float:
        return sum(self.requested_durations)

    def sleep(self, seconds: float) -> None:
        self.requested_durations.append(seconds)
