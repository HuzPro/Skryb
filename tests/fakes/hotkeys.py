from collections.abc import Callable

from voicetotext.hotkeys.listener import HotkeyListener


class FakeHotkeyListener(HotkeyListener):
    """In-memory HotkeyListener that counts its lifecycle calls."""

    def __init__(self) -> None:
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.pause_calls: int = 0
        self.resume_calls: int = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1


class FakeClock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeTimer:
    def __init__(self, cancelled: list[bool]) -> None:
        self._cancelled = cancelled

    def start(self) -> None:
        pass

    def cancel(self) -> None:
        self._cancelled[0] = True


class FakeTimers:
    """Timers that only fire when the test says their time has come."""

    def __init__(self) -> None:
        self.pending: list[tuple[float, list[bool], Callable[[], None]]] = []

    def make(self, seconds: float, action: Callable[[], None]) -> FakeTimer:
        cancelled = [False]
        self.pending.append((seconds, cancelled, action))
        return FakeTimer(cancelled)

    def fire_all(self) -> None:
        pending, self.pending = self.pending, []
        for _seconds, cancelled, action in pending:
            if not cancelled[0]:
                action()
