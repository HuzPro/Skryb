"""Tests for :class:`ThreadedActionDispatcher`.

The dispatcher runs each action on a daemon thread. Without an explicit
top-level catch, an unhandled exception inside the action prints to
stderr via Python's default ``threading.excepthook`` - easy to miss
when stderr is the terminal and the user is reading it casually.
Wrapping with try/except + ``logger.exception`` ensures the failure
also lands in the persistent debug log.
"""

import logging
import threading
import time

from voicetotext.core.action_dispatcher import ThreadedActionDispatcher


def _wait_for_thread_count_baseline(timeout: float = 1.0) -> None:
    """Wait briefly for the dispatched daemon thread to finish."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if threading.active_count() <= 2:  # main + pytest
            return
        time.sleep(0.01)


class TestThreadedActionDispatcherDispatch:
    def test_should_run_action_on_a_daemon_thread(self) -> None:
        ran = threading.Event()

        ThreadedActionDispatcher().dispatch(ran.set)

        assert ran.wait(timeout=1.0) is True

    def test_should_log_unhandled_exceptions_via_logger(self) -> None:
        ran = threading.Event()
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        # Attach our own handler directly to the dispatcher's logger;
        # caplog captures via the root handler, but the ``voicetotext``
        # subtree has ``propagate=False`` after ``setup_debug_logger``
        # has been called by an earlier test, which makes caplog blind
        # to it.
        dispatcher_logger = logging.getLogger(
            "voicetotext.core.action_dispatcher"
        )
        capture = _Capture(level=logging.ERROR)
        dispatcher_logger.addHandler(capture)

        def boom() -> None:
            try:
                raise RuntimeError("kaboom")
            finally:
                ran.set()

        try:
            ThreadedActionDispatcher().dispatch(boom)
            assert ran.wait(timeout=1.0) is True
            time.sleep(0.05)  # let the wrapping try/except log
        finally:
            dispatcher_logger.removeHandler(capture)

        messages = [r.getMessage() for r in records]
        assert any("dispatched action failed" in m for m in messages), messages

    def test_should_not_propagate_action_exceptions(self) -> None:
        # The dispatcher must never let an action crash kill the dispatcher
        # or its caller. Subsequent dispatches must still work after a
        # failed one.
        def boom() -> None:
            raise RuntimeError("first")

        ran = threading.Event()
        dispatcher = ThreadedActionDispatcher()

        dispatcher.dispatch(boom)
        _wait_for_thread_count_baseline()
        dispatcher.dispatch(ran.set)

        assert ran.wait(timeout=1.0) is True
