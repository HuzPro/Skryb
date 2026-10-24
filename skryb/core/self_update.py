"""Restart the app onto newer code, merged on GitHub or changed in this folder.

It runs once at start-up and again whenever the app is launched while it is
already running. A restart never cuts off a dictation: it waits for the
dictation to finish.
"""

import logging
import time
from collections.abc import Callable

_logger = logging.getLogger(__name__)

_IDLE_POLL_SECONDS = 0.5

CHECKING = "Checking for newer code..."
UP_TO_DATE = "Already running the latest code."
NOT_PULLED = "Running this folder's code. Didn't pull from GitHub: {reason}."
AFTER_DICTATION = "Restarting into the latest code after this dictation."


class SelfUpdate:
    def __init__(
        self,
        *,
        pull: Callable[[], str | None],
        source_changed: Callable[[], bool],
        is_busy: Callable[[], bool],
        restart: Callable[[], None],
        tell: Callable[[str], None],
        on_gui_thread: Callable[[Callable[[], None]], None],
        in_background: Callable[[Callable[[], None]], None],
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._pull = pull
        self._source_changed = source_changed
        self._is_busy = is_busy
        self._restart = restart
        self._tell = tell
        self._on_gui_thread = on_gui_thread
        self._in_background = in_background
        self._sleep = sleep
        self._checking = False

    def check(self, *, asked: bool) -> None:
        """Pull, then restart if the code changed. *asked* means the icon was clicked, which always gets an answer."""
        if self._checking:
            return
        self._checking = True
        if asked:
            self._tell(CHECKING)
        self._in_background(lambda: self._check(asked))

    def _check(self, asked: bool) -> None:
        reason = self._pull()
        if reason is not None:
            _logger.info("Not updating from GitHub: %s.", reason)
        if self._source_changed():
            _logger.info("Newer code is on disk; restarting onto it.")
            self._restart_when_idle()
            return
        self._checking = False
        if asked:
            self._tell(UP_TO_DATE if reason is None else NOT_PULLED.format(reason=reason))

    def _restart_when_idle(self) -> None:
        if self._is_busy():
            self._tell(AFTER_DICTATION)
            while self._is_busy():
                self._sleep(_IDLE_POLL_SECONDS)
        self._on_gui_thread(self._restart_unless_busy)

    def _restart_unless_busy(self) -> None:
        # A dictation can start between the last look and this running on
        # the Tk thread.
        if self._is_busy():
            self._in_background(self._restart_when_idle)
            return
        self._restart()
