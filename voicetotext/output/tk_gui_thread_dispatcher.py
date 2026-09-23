import logging
import queue
import time
import tkinter
from collections.abc import Callable

from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher


_logger = logging.getLogger(__name__)
_DRAIN_INTERVAL_MS = 15
_PUMP_INTERVAL_SECONDS = 0.01


class TkGuiThreadDispatcher(GuiThreadDispatcher):
    """Tk-backed dispatcher: a thread-safe queue drained on the GUI thread.

    ``dispatch`` enqueues from any thread; a periodic ``after`` poller on
    the Tk root drains the queue and runs each action on the main (GUI)
    thread, where Win32 clipboard / SendInput calls are safe. ``after``
    callbacks are not assumed thread-safe, so producers only touch the
    queue - never the widget tree.
    """

    def __init__(self, root: tkinter.Misc) -> None:
        self._root = root
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._after_id = self._root.after(_DRAIN_INTERVAL_MS, self._drain)

    def dispatch(self, action: Callable[[], None]) -> None:
        self._queue.put(action)

    def sleep(self, seconds: float) -> None:
        """Wait while Tk keeps handling events. GUI thread only.

        Our own settings window reads an injected Ctrl+V from this thread, so
        blocking it would leave the paste queued until after the clipboard is
        restored. No drain is scheduled while an action runs, so pumping here
        cannot start another queued action.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._root.update()
            time.sleep(_PUMP_INTERVAL_SECONDS)

    def cancel(self) -> None:
        """Stop the drain poller (idempotent). Used on shutdown/teardown."""
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
            self._after_id = None

    def _drain(self) -> None:
        while True:
            try:
                action = self._queue.get_nowait()
            except queue.Empty:
                break
            self._invoke(action)
        self._after_id = self._root.after(_DRAIN_INTERVAL_MS, self._drain)

    def _invoke(self, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception:
            _logger.exception("GUI-thread action raised; swallowed.")
