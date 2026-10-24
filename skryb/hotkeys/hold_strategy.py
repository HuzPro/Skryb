import functools
import threading
import time
from collections.abc import Callable
from typing import Protocol

from skryb.hotkeys.keys import HotkeyKey
from skryb.hotkeys.strategy import ComboMatcher, HotkeyStrategy

DOUBLE_TAP_SECONDS = 0.4


class Timer(Protocol):
    def start(self) -> None: ...
    def cancel(self) -> None: ...


MakeTimer = Callable[[float, Callable[[], None]], Timer]


def _daemon_timer(seconds: float, action: Callable[[], None]) -> Timer:
    timer = threading.Timer(seconds, action)
    timer.daemon = True
    return timer


def _nothing() -> None:
    return None


class HoldStrategy(HotkeyStrategy):
    """Hold the combo to talk, release it to stop.

    With ``double_tap_locks`` two full presses within
    :data:`DOUBLE_TAP_SECONDS` lock it on hands-free. A first press
    released that quickly may be the start of a double tap, so its stop
    waits for the window to close; a press held longer is an ordinary hold
    and stops the moment it is released.

    The press after a lock sends ``on_activate``, the same toggle request a
    Toggle shortcut sends, rather than ``on_deactivate``: if the recording
    was already ended some other way (cancel, the pill, the length cap)
    that press starts a new one instead of being wasted.

    The pending stop fires on a timer thread, hence the lock.
    """

    def __init__(
        self,
        keys: tuple[HotkeyKey, ...],
        double_tap_locks: bool = False,
        clock: Callable[[], float] = time.monotonic,
        make_timer: MakeTimer = _daemon_timer,
    ) -> None:
        self._matcher = ComboMatcher(keys)
        self._double_tap_locks = double_tap_locks
        self._clock = clock
        self._make_timer = make_timer
        self._guard = threading.Lock()
        self._active = False
        self._locked = False
        self._pressed_at = 0.0
        self._pending_stop: Timer | None = None
        self._taps = 0
        self._on_activate: Callable[[], None] = _nothing
        self._on_deactivate: Callable[[], None] = _nothing

    def set_callbacks(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate

    def on_key_press(self, key: HotkeyKey) -> bool:
        with self._guard:
            if not self._matcher.press(key):
                return False
            if self._locked:
                self._locked = False
                self._active = False
                self._on_activate()
            elif self._pending_stop is not None:
                self._pending_stop.cancel()
                self._pending_stop = None
                self._locked = True
            else:
                self._start()
            return True

    def on_key_release(self, key: HotkeyKey) -> None:
        with self._guard:
            if not self._matcher.release(key) or not self._active or self._locked:
                return
            held_for = self._clock() - self._pressed_at
            if self._double_tap_locks and held_for <= DOUBLE_TAP_SECONDS:
                self._wait_for_second_tap(DOUBLE_TAP_SECONDS - held_for)
            else:
                self._stop()

    def _start(self) -> None:
        self._active = True
        self._pressed_at = self._clock()
        self._on_activate()

    def _stop(self) -> None:
        self._active = False
        self._on_deactivate()

    def _wait_for_second_tap(self, seconds: float) -> None:
        self._taps += 1
        self._pending_stop = self._make_timer(
            seconds, functools.partial(self._stop_unless_tapped_again, self._taps),
        )
        self._pending_stop.start()

    def _stop_unless_tapped_again(self, tap: int) -> None:
        with self._guard:
            if self._pending_stop is None or tap != self._taps:
                return
            self._pending_stop = None
            self._stop()
