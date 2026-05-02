"""Tests for :class:`QtGuiThreadDispatcher`.

Verifies the same-thread direct-invoke path and the cross-thread
queued-invoke path. The latter is what fixes the Win32-from-daemon-
thread crash, so we exercise it with a real worker thread.
"""

import threading

import pytest

from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher
from voicetotext.output.qt_gui_thread_dispatcher import QtGuiThreadDispatcher


@pytest.fixture
def dispatcher(qtbot):
    return QtGuiThreadDispatcher()


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
