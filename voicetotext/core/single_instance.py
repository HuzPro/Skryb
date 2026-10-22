"""One copy of the app per Windows session, and what a second launch does instead.

A second launch asks the copy already running to answer for it, which is how
clicking the icon again checks for newer code. A copy that never answers is
stuck, so the second launch ends it and takes its place.
"""

import ctypes
import ctypes.wintypes as wintypes
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


_ERROR_ALREADY_EXISTS = 183
_WAIT_OBJECT_0 = 0
_PROCESS_TERMINATE = 0x0001
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_MUTEX_NAME = "Local\\VoiceToText"
# A second launch sets the first event, and the copy running sets the second
# to answer it.
LAUNCH_EVENT_NAMES = ("Local\\VoiceToText-launched-again", "Local\\VoiceToText-launch-answered")
_RETRY_SECONDS = 0.1
_POLL_MS = 250
_TASKKILL_TIMEOUT_SECONDS = 15


class KernelMutexApi(Protocol):
    def create_mutex(self, name: str) -> int: ...
    def last_error(self) -> int: ...
    def close_handle(self, handle: int) -> None: ...


class KernelEventApi(Protocol):
    def create_event(self, name: str) -> int: ...
    def set_event(self, handle: int) -> None: ...
    def reset_event(self, handle: int) -> None: ...
    def wait(self, handle: int, milliseconds: int) -> bool: ...


class ProcessApi(Protocol):
    def process_started(self, pid: int) -> int | None: ...
    def terminate(self, pid: int, started: int) -> None: ...


class Win32KernelMutexApi:
    def __init__(self) -> None:
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR,
        ]
        self._kernel32.CreateMutexW.restype = wintypes.HANDLE
        self._kernel32.CreateEventW.argtypes = [
            ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR,
        ]
        self._kernel32.CreateEventW.restype = wintypes.HANDLE
        self._kernel32.SetEvent.argtypes = [wintypes.HANDLE]
        self._kernel32.SetEvent.restype = wintypes.BOOL
        self._kernel32.ResetEvent.argtypes = [wintypes.HANDLE]
        self._kernel32.ResetEvent.restype = wintypes.BOOL
        self._kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._kernel32.WaitForSingleObject.restype = wintypes.DWORD
        self._kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
            ctypes.POINTER(wintypes.FILETIME),
        ] * 4
        self._kernel32.GetProcessTimes.restype = wintypes.BOOL
        self._kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateProcess.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL

    def create_mutex(self, name: str) -> int:
        handle = self._kernel32.CreateMutexW(None, False, name)
        return int(handle or 0)

    def create_event(self, name: str) -> int:
        # Auto-reset: one successful wait consumes the signal.
        handle = self._kernel32.CreateEventW(None, False, False, name)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateEventW failed")
        return int(handle)

    def set_event(self, handle: int) -> None:
        self._kernel32.SetEvent(handle)

    def reset_event(self, handle: int) -> None:
        self._kernel32.ResetEvent(handle)

    def wait(self, handle: int, milliseconds: int) -> bool:
        return bool(self._kernel32.WaitForSingleObject(handle, milliseconds) == _WAIT_OBJECT_0)

    def process_started(self, pid: int) -> int | None:
        """When *pid* was created, in FILETIME ticks, or None when it can't be opened."""
        handle = self._kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            return self._creation_time(handle)
        finally:
            self._kernel32.CloseHandle(handle)

    def terminate(self, pid: int, started: int) -> None:
        """End *pid* if it is still the process created at *started*."""
        handle = self._kernel32.OpenProcess(
            _PROCESS_TERMINATE | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
        )
        if not handle:
            return
        try:
            self._terminate_if_started_at(handle, started)
        finally:
            self._kernel32.CloseHandle(handle)

    def _terminate_if_started_at(self, handle: int, started: int) -> None:
        if self._creation_time(handle) == started:
            self._kernel32.TerminateProcess(handle, 1)

    def _creation_time(self, handle: int) -> int | None:
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not self._kernel32.GetProcessTimes(
            handle, ctypes.byref(created), ctypes.byref(exited),
            ctypes.byref(kernel), ctypes.byref(user),
        ):
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime

    def last_error(self) -> int:
        return int(ctypes.get_last_error())

    def close_handle(self, handle: int) -> None:
        self._kernel32.CloseHandle(handle)


class SingleInstanceMutex:
    def __init__(
        self,
        api: KernelMutexApi | None = None,
        name: str = _MUTEX_NAME,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api = api or Win32KernelMutexApi()
        self._name = name
        self._sleep = sleep
        self._clock = clock
        self._handle = 0

    def acquire(self, wait_seconds: float = 0.0) -> bool:
        """Take the lock, waiting up to *wait_seconds* for the copy holding it to exit."""
        deadline = self._clock() + wait_seconds
        while not self._try_acquire():
            if self._clock() >= deadline:
                return False
            self._sleep(_RETRY_SECONDS)
        return True

    def _try_acquire(self) -> bool:
        self._handle = self._api.create_mutex(self._name)
        if not self._handle:
            raise OSError("CreateMutexW failed")
        if self._api.last_error() == _ERROR_ALREADY_EXISTS:
            self.release()
            return False
        return True

    def release(self) -> None:
        if self._handle:
            self._api.close_handle(self._handle)
            self._handle = 0


class LaunchSignal:
    """How a second launch asks the copy already running to answer for it."""

    def __init__(
        self,
        api: KernelEventApi | None = None,
        names: tuple[str, str] = LAUNCH_EVENT_NAMES,
    ) -> None:
        self._api = api or Win32KernelMutexApi()
        self._asked = self._api.create_event(names[0])
        self._answered = self._api.create_event(names[1])

    def ask(self, timeout_seconds: float) -> bool:
        """From the second launch: True once the copy running has answered."""
        self._api.reset_event(self._answered)
        self._api.set_event(self._asked)
        if self._api.wait(self._answered, round(timeout_seconds * 1000)):
            return True
        # Unanswered, so the copy that replaces the stuck one must not find
        # this request waiting for it.
        self._api.reset_event(self._asked)
        return False

    def answer(self) -> bool:
        """From the copy running: True, and answered, when a second launch is asking."""
        if not self._api.wait(self._asked, 0):
            return False
        self._api.set_event(self._answered)
        return True


class TkAfter(Protocol):
    def after(self, ms: int, func: Callable[[], None]) -> str: ...


def _ignore() -> None:
    return None


class SecondLaunches:
    """Answers a second launch of the app from the Tk thread.

    Answering on the Tk thread, not a thread of its own, is what shows the
    second launch that this copy is alive. A copy whose Tk thread is stuck
    never answers, and the second launch replaces it.
    """

    def __init__(self, root: TkAfter, signal: LaunchSignal) -> None:
        self.on_launch: Callable[[], None] = _ignore
        self._root = root
        self._signal = signal
        self._root.after(_POLL_MS, self._poll)

    def _poll(self) -> None:
        self._root.after(_POLL_MS, self._poll)
        if self._signal.answer():
            self.on_launch()


class RunningCopy:
    """Which process holds the lock, kept so a later launch can end it if it is stuck."""

    def __init__(self, path: Path, api: ProcessApi | None = None) -> None:
        self._path = path
        self._api = api or Win32KernelMutexApi()

    def record(self, pid: int | None = None) -> None:
        pid = os.getpid() if pid is None else pid
        try:
            self._write(pid)
        except OSError:
            # Only costs a later launch the means to end this copy if it hangs.
            pass

    def _write(self, pid: int) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(f"{pid} {self._api.process_started(pid)}", encoding="ascii")

    def end(self) -> bool:
        """End the recorded copy and what it started; False when it is not the same process."""
        recorded = self._read()
        if recorded is None:
            return False
        pid, started = recorded
        # Matching the start time means a pid Windows has since reused for
        # some other program is never ended.
        if self._api.process_started(pid) != started:
            return False
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
            creationflags=_CREATE_NO_WINDOW,
            timeout=_TASKKILL_TIMEOUT_SECONDS,
        )
        # taskkill ends what the copy started, but not a copy stuck in its
        # own exit: it reports "no running instance" and leaves it running.
        self._api.terminate(pid, started)
        return True

    def _read(self) -> tuple[int, int] | None:
        try:
            return _parse_record(self._path.read_text(encoding="ascii"))
        except (OSError, ValueError, UnicodeError):
            return None


def _parse_record(text: str) -> tuple[int, int]:
    pid, started = text.split()
    return int(pid), int(started)
