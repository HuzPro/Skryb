"""Tests for :class:`QtGuiThreadDispatcher`.

Verifies the same-thread direct-invoke path and the cross-thread
queued-invoke path. The latter is what fixes the Win32-from-daemon-
thread crash, so we exercise it with a real worker thread.
"""

import logging
import threading

import pytest

from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.output.qt_gui_thread_dispatcher import QtGuiThreadDispatcher


_DISPATCHER_LOGGER = "voicetotext.output.qt_gui_thread_dispatcher"


@pytest.fixture
def dispatcher(qtbot):
    return QtGuiThreadDispatcher()


@pytest.fixture()
def log_capture(caplog: pytest.LogCaptureFixture):
    # Some other test in the suite mutates logging propagation, so caplog
    # cannot reliably capture via the root handler. Attach explicitly.
    logger = logging.getLogger(_DISPATCHER_LOGGER)
    logger.addHandler(caplog.handler)
    caplog.handler.setLevel(logging.DEBUG)
    yield caplog
    logger.removeHandler(caplog.handler)


class TestQtGuiThreadDispatcherInterface:
    def test_should_implement_gui_thread_dispatcher(self, dispatcher) -> None:
        assert isinstance(dispatcher, GuiThreadDispatcher)


class TestQtGuiThreadDispatcherSameThread:
    def test_should_invoke_action_when_dispatched_from_gui_thread(
        self, qtbot, dispatcher,
    ) -> None:
        ran = threading.Event()

        dispatcher.dispatch(ran.set)
        # Drain pending Qt events so the queued action runs even when
        # called from the GUI thread (AutoConnection picks DirectConn
        # same-thread, but emit -> connected slot is still subject to
        # signal delivery order).
        qtbot.wait(50)

        assert ran.is_set() is True


class TestQtGuiThreadDispatcherCrossThread:
    def test_should_invoke_action_on_gui_thread_when_dispatched_from_worker(
        self, qtbot, dispatcher,
    ) -> None:
        gui_thread = threading.current_thread()
        observed_thread: list[threading.Thread] = []
        invoked = threading.Event()

        def action() -> None:
            observed_thread.append(threading.current_thread())
            invoked.set()

        def worker() -> None:
            dispatcher.dispatch(action)

        worker_thread = threading.Thread(target=worker, daemon=True)
        worker_thread.start()
        worker_thread.join(timeout=1.0)

        # The action should NOT yet have run — it's queued onto the
        # GUI event loop, which only spins when we yield via qtbot.wait.
        assert invoked.is_set() is False

        qtbot.waitUntil(invoked.is_set, timeout=1000)

        assert observed_thread == [gui_thread]

    def test_should_serialize_multiple_cross_thread_dispatches(
        self, qtbot, dispatcher,
    ) -> None:
        results: list[int] = []
        done = threading.Event()

        def make_action(value: int):
            def action() -> None:
                results.append(value)
                if value == 4:
                    done.set()
            return action

        def worker() -> None:
            for i in range(5):
                dispatcher.dispatch(make_action(i))

        threading.Thread(target=worker, daemon=True).start()

        qtbot.waitUntil(done.is_set, timeout=1000)

        assert results == [0, 1, 2, 3, 4]


class TestQtGuiThreadDispatcherExceptionSafety:
    """An exception escaping a Qt slot can take down the event loop —
    PySide6 may invoke ``sys.excepthook`` and on some Qt builds this
    terminates the process. The clipboard-preservation path on the GUI
    thread is the immediate motivator: a Win32 ctypes call with a bad
    handle could raise (or even segfault, but at least Python-level
    exceptions must be contained).
    """

    def test_should_log_and_swallow_action_exception(
        self, qtbot, dispatcher, log_capture,
    ) -> None:
        def boom() -> None:
            raise RuntimeError("simulated paste failure")

        dispatcher.dispatch(boom)
        qtbot.wait(50)

        assert "simulated paste failure" in log_capture.text

    def test_should_still_invoke_subsequent_actions_after_exception(
        self, qtbot, dispatcher,
    ) -> None:
        ran_after_boom = threading.Event()

        dispatcher.dispatch(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        dispatcher.dispatch(ran_after_boom.set)

        qtbot.waitUntil(ran_after_boom.is_set, timeout=1000)
