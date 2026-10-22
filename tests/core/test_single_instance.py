import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from voicetotext.core.single_instance import (
    LaunchSignal,
    RunningCopy,
    SecondLaunches,
    SingleInstanceMutex,
    Win32KernelMutexApi,
)

_ASKED, _ANSWERED = "Local\\VoiceToText-launched-again", "Local\\VoiceToText-launch-answered"


class FakeMutexApi:
    def __init__(self, error: int = 0) -> None:
        self.error = error
        self.closed: list[int] = []

    def create_mutex(self, name: str) -> int:
        assert name == "Local\\VoiceToText"
        return 42

    def last_error(self) -> int:
        return self.error

    def close_handle(self, handle: int) -> None:
        self.closed.append(handle)


class TestSingleInstanceMutex:
    def test_should_retain_handle_until_release(self) -> None:
        api = FakeMutexApi()
        mutex = SingleInstanceMutex(api)

        assert mutex.acquire() is True
        assert api.closed == []

        mutex.release()
        assert api.closed == [42]

    def test_should_close_conflicting_handle_and_reject_second_instance(self) -> None:
        api = FakeMutexApi(error=183)
        mutex = SingleInstanceMutex(api)

        assert mutex.acquire() is False
        assert api.closed == [42]


class FlakyMutexApi(FakeMutexApi):
    """The other copy holds the lock for the first *held_for* looks."""

    def __init__(self, held_for: int) -> None:
        super().__init__()
        self.held_for = held_for

    def last_error(self) -> int:
        self.held_for -= 1
        return 183 if self.held_for >= 0 else 0


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class TestWaitingForTheLock:
    def test_a_restarting_copy_gets_the_lock_once_the_copy_it_replaces_exits(self) -> None:
        clock = FakeClock()
        mutex = SingleInstanceMutex(FlakyMutexApi(held_for=25), sleep=clock.sleep, clock=clock)

        assert mutex.acquire(wait_seconds=30.0) is True
        assert clock.now == pytest.approx(2.5)

    def test_a_copy_that_never_exits_leaves_the_lock_where_it_is(self) -> None:
        clock = FakeClock()
        api = FlakyMutexApi(held_for=10_000)
        mutex = SingleInstanceMutex(api, sleep=clock.sleep, clock=clock)

        assert mutex.acquire(wait_seconds=30.0) is False
        assert clock.now == pytest.approx(30.0, abs=0.11)
        # Every look closed the handle it opened.
        assert len(api.closed) == 10_000 - api.held_for

    def test_without_a_wait_it_looks_once(self) -> None:
        api = FlakyMutexApi(held_for=1)

        assert SingleInstanceMutex(api).acquire() is False
        assert api.held_for == 0


class FakeEvents:
    """Named auto-reset events shared by every LaunchSignal made from it."""

    def __init__(self) -> None:
        self.signalled: dict[str, bool] = {}
        self._names: dict[int, str] = {}

    def create_event(self, name: str) -> int:
        handle = len(self._names) + 1
        self._names[handle] = name
        self.signalled.setdefault(name, False)
        return handle

    def set_event(self, handle: int) -> None:
        self.signalled[self._names[handle]] = True

    def reset_event(self, handle: int) -> None:
        self.signalled[self._names[handle]] = False

    def wait(self, handle: int, milliseconds: int) -> bool:
        name = self._names[handle]
        if self.signalled[name]:
            self.signalled[name] = False
            return True
        return False


class AnsweringEvents(FakeEvents):
    """The copy running answers while the second launch waits."""

    def __init__(self) -> None:
        super().__init__()
        self.running: LaunchSignal | None = None

    def wait(self, handle: int, milliseconds: int) -> bool:
        if milliseconds and self.running is not None:
            self.running.answer()
        return super().wait(handle, milliseconds)


class TestLaunchSignal:
    def test_a_second_launch_is_answered_by_the_copy_running(self) -> None:
        events = AnsweringEvents()
        events.running = LaunchSignal(events)

        assert LaunchSignal(events).ask(timeout_seconds=20) is True

    def test_the_copy_running_sees_nothing_when_nobody_asked(self) -> None:
        assert LaunchSignal(FakeEvents()).answer() is False

    def test_a_request_is_answered_once(self) -> None:
        events = FakeEvents()
        running = LaunchSignal(events)
        events.signalled[_ASKED] = True

        assert running.answer() is True
        assert running.answer() is False

    def test_an_unanswered_request_is_withdrawn_for_the_copy_that_replaces_a_stuck_one(
        self,
    ) -> None:
        events = FakeEvents()

        assert LaunchSignal(events).ask(timeout_seconds=20) is False
        assert LaunchSignal(events).answer() is False

    def test_a_stale_answer_does_not_count_for_a_new_request(self) -> None:
        events = FakeEvents()
        events.signalled[_ANSWERED] = True

        assert LaunchSignal(events).ask(timeout_seconds=20) is False


class FakeRoot:
    def __init__(self) -> None:
        self.scheduled: list = []

    def after(self, _ms: int, func) -> str:
        self.scheduled.append(func)
        return str(len(self.scheduled))

    def run_next(self) -> None:
        self.scheduled.pop(0)()


class TestSecondLaunches:
    def test_a_second_launch_reaches_the_app_from_the_tk_poll(self) -> None:
        events = FakeEvents()
        root = FakeRoot()
        launches = SecondLaunches(root, LaunchSignal(events))
        heard: list[str] = []
        launches.on_launch = lambda: heard.append("launched again")

        root.run_next()
        events.signalled[_ASKED] = True
        root.run_next()

        assert heard == ["launched again"]
        assert events.signalled[_ANSWERED] is True

    def test_the_poll_keeps_going_after_the_app_fails_to_handle_a_launch(self) -> None:
        events = FakeEvents()
        root = FakeRoot()
        launches = SecondLaunches(root, LaunchSignal(events))
        launches.on_launch = lambda: 1 / 0
        events.signalled[_ASKED] = True

        with pytest.raises(ZeroDivisionError):
            root.run_next()

        assert len(root.scheduled) == 1

    def test_before_the_app_is_built_a_launch_is_answered_and_nothing_else(self) -> None:
        events = FakeEvents()
        root = FakeRoot()
        SecondLaunches(root, LaunchSignal(events))
        events.signalled[_ASKED] = True

        root.run_next()

        assert events.signalled[_ANSWERED] is True


class FakeProcesses:
    def __init__(self, started: dict[int, int]) -> None:
        self.started = started
        self.terminated: list[tuple[int, int]] = []

    def process_started(self, pid: int) -> int | None:
        return self.started.get(pid)

    def terminate(self, pid: int, started: int) -> None:
        self.terminated.append((pid, started))


class TestRunningCopy:
    def test_a_record_names_the_process_and_when_it_started(self, tmp_path: Path) -> None:
        path = tmp_path / "VoiceToText" / "running.pid"

        RunningCopy(path, FakeProcesses({4321: 133_000_000_000_000_000})).record(4321)

        assert path.read_text(encoding="ascii") == "4321 133000000000000000"

    def test_a_record_that_cannot_be_written_does_not_stop_the_app(self, tmp_path: Path) -> None:
        blocker = tmp_path / "VoiceToText"
        blocker.write_text("a file where the folder should be", encoding="utf-8")

        RunningCopy(blocker / "running.pid", FakeProcesses({1: 1})).record(1)

    @pytest.mark.parametrize(
        "text",
        ["", "   ", "4321", "4321 not-a-time", "four 1", "4321 1 extra", "٣٤ 1"],
    )
    def test_a_missing_or_garbled_record_ends_nothing(self, tmp_path: Path, text: str) -> None:
        path = tmp_path / "running.pid"
        path.write_text(text, encoding="utf-8")

        with patch("voicetotext.core.single_instance.subprocess.run") as run:
            assert RunningCopy(path, FakeProcesses({4321: 1})).end() is False

        run.assert_not_called()

    def test_no_record_at_all_ends_nothing(self, tmp_path: Path) -> None:
        with patch("voicetotext.core.single_instance.subprocess.run") as run:
            assert RunningCopy(tmp_path / "running.pid", FakeProcesses({})).end() is False

        run.assert_not_called()

    def test_a_pid_windows_has_reused_for_another_program_is_never_ended(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "running.pid"
        path.write_text("4321 133000000000000000", encoding="ascii")

        processes = FakeProcesses({4321: 133000000099999999})

        with patch("voicetotext.core.single_instance.subprocess.run") as run:
            ended = RunningCopy(path, processes).end()

        assert ended is False
        run.assert_not_called()
        assert processes.terminated == []

    def test_a_copy_taskkill_cannot_end_is_also_ended_directly(self, tmp_path: Path) -> None:
        path = tmp_path / "running.pid"
        path.write_text("4321 133000000000000000", encoding="ascii")
        processes = FakeProcesses({4321: 133000000000000000})

        with patch("voicetotext.core.single_instance.subprocess.run") as run:
            assert RunningCopy(path, processes).end() is True

        run.assert_called_once()
        assert processes.terminated == [(4321, 133000000000000000)]


windows_only = pytest.mark.skipif(sys.platform != "win32", reason="real Win32 kernel objects")


def _sleeper() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@windows_only
class TestRealKernelObjects:
    """Unique names, so nothing here can reach the copy of the app actually running."""

    def test_two_launches_talk_through_the_named_events_across_threads(self) -> None:
        names = (f"Local\\vtt-test-{uuid.uuid4()}", f"Local\\vtt-test-{uuid.uuid4()}")
        running = LaunchSignal(names=names)
        answered = threading.Event()

        def answer_soon() -> None:
            while not running.answer():
                time.sleep(0.01)
            answered.set()

        threading.Thread(target=answer_soon, daemon=True).start()

        assert LaunchSignal(names=names).ask(timeout_seconds=5) is True
        assert answered.wait(5)

    def test_an_unanswered_launch_gives_up_after_its_timeout(self) -> None:
        names = (f"Local\\vtt-test-{uuid.uuid4()}", f"Local\\vtt-test-{uuid.uuid4()}")
        LaunchSignal(names=names)
        started = time.monotonic()

        assert LaunchSignal(names=names).ask(timeout_seconds=0.3) is False
        assert time.monotonic() - started >= 0.25

    def test_the_lock_comes_free_when_the_copy_holding_it_lets_go(self) -> None:
        name = f"Local\\vtt-test-{uuid.uuid4()}"
        holder = SingleInstanceMutex(name=name)
        assert holder.acquire() is True
        threading.Timer(0.3, holder.release).start()

        assert SingleInstanceMutex(name=name).acquire(wait_seconds=5) is True

    def test_a_stuck_copy_is_ended(self, tmp_path: Path) -> None:
        stuck = _sleeper()
        copy = RunningCopy(tmp_path / "running.pid")
        copy.record(stuck.pid)

        assert copy.end() is True
        assert stuck.wait(timeout=10) is not None

    def test_a_live_process_that_is_not_the_recorded_one_is_left_running(
        self, tmp_path: Path,
    ) -> None:
        bystander = _sleeper()
        try:
            path = tmp_path / "running.pid"
            path.write_text(f"{bystander.pid} 1", encoding="ascii")

            assert RunningCopy(path).end() is False
            Win32KernelMutexApi().terminate(bystander.pid, 1)
            with pytest.raises(subprocess.TimeoutExpired):
                bystander.wait(timeout=0.5)
        finally:
            bystander.kill()
            bystander.wait(timeout=10)
