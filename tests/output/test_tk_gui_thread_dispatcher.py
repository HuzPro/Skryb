"""Tests for :class:`TkGuiThreadDispatcher`.

Verifies that actions enqueued from any thread run on the Tk main thread
(where Win32 clipboard / SendInput calls are safe), in order, and that a
failing action is logged and swallowed without stopping the drain.
"""

import logging
import threading
import time

import pytest

from skryb.output.gui_thread_dispatcher import GuiThreadDispatcher
from skryb.output.tk_gui_thread_dispatcher import TkGuiThreadDispatcher


_DISPATCHER_LOGGER = "skryb.output.tk_gui_thread_dispatcher"


@pytest.fixture
def root(tk_root):
    # Reuse the session-scoped root from conftest. Repeatedly creating and
    # destroying Tk() in a process that also runs the Qt suite corrupts
    # tkinter's tcl/tk library path; one long-lived root sidesteps it.
    return tk_root


@pytest.fixture
def dispatcher(root):
    dispatcher = TkGuiThreadDispatcher(root)
    yield dispatcher
    dispatcher.cancel()


def _pump_until(root, predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        root.update()
        time.sleep(0.005)
    root.update()


class TestTkGuiThreadDispatcherInterface:
    def test_should_implement_gui_thread_dispatcher(self, dispatcher) -> None:
        assert isinstance(dispatcher, GuiThreadDispatcher)


class TestTkGuiThreadDispatcherDrain:
    def test_should_invoke_action_dispatched_from_main_thread(
        self, root, dispatcher,
    ) -> None:
        ran = threading.Event()

        dispatcher.dispatch(ran.set)
        _pump_until(root, ran.is_set)

        assert ran.is_set() is True

    def test_should_invoke_action_on_main_thread_when_dispatched_from_worker(
        self, root, dispatcher,
    ) -> None:
        main_thread = threading.current_thread()
        observed: list[threading.Thread] = []
        invoked = threading.Event()

        def action() -> None:
            observed.append(threading.current_thread())
            invoked.set()

        threading.Thread(
            target=lambda: dispatcher.dispatch(action), daemon=True,
        ).start()
        _pump_until(root, invoked.is_set)

        assert observed == [main_thread]

    def test_should_run_actions_in_dispatch_order(self, root, dispatcher) -> None:
        results: list[int] = []
        done = threading.Event()

        def worker() -> None:
            for value in range(5):
                dispatcher.dispatch(lambda v=value: results.append(v))
            dispatcher.dispatch(done.set)

        threading.Thread(target=worker, daemon=True).start()
        _pump_until(root, done.is_set)

        assert results == [0, 1, 2, 3, 4]


class TestTkGuiThreadDispatcherSleep:
    def test_sleep_returns_on_time_while_a_frame_loop_keeps_a_timer_always_due(
        self, root, dispatcher,
    ) -> None:
        # Like the pill: each frame reschedules itself and is due again at
        # once. The loop gives up after 1.5 s so a regression fails, not hangs.
        give_up_at = time.monotonic() + 1.5

        def frame() -> None:
            time.sleep(0.002)
            if time.monotonic() < give_up_at:
                root.after(0, frame)

        root.after(0, frame)
        started = time.monotonic()
        dispatcher.sleep(0.05)
        took = time.monotonic() - started
        while time.monotonic() < give_up_at + 0.1:
            root.update()

        assert took < 0.5

    def test_a_frame_loop_keeps_drawing_every_few_milliseconds_while_a_paste_waits(
        self, root, dispatcher,
    ) -> None:
        # The pill reschedules itself 1 ms out after each frame. A 100 ms
        # clipboard settle that dozed 10 ms between events let it draw only
        # every 10 ms or so, which showed as a stutter on every dictation.
        drawn: list[float] = []
        stopped = False

        def frame() -> None:
            drawn.append(time.monotonic())
            if not stopped:
                root.after(1, frame)

        root.after(1, frame)
        dispatcher.sleep(0.1)
        # The root is shared: a loop left running would starve later tests.
        stopped = True
        root.update()
        gaps = [after - before for before, after in zip(drawn, drawn[1:])]

        assert len(drawn) > 25
        assert sorted(gaps)[len(gaps) // 2] < 0.006

    def test_waiting_off_the_gui_thread_never_touches_tk(self, dispatcher) -> None:
        failures: list[BaseException] = []

        def wait() -> None:
            try:
                dispatcher.sleep(0.02)
            except BaseException as exc:  # noqa: BLE001
                failures.append(exc)

        worker = threading.Thread(target=wait)
        worker.start()
        worker.join(timeout=2)

        assert failures == []
        assert not worker.is_alive()


class TestTkGuiThreadDispatcherExceptionSafety:
    @pytest.fixture()
    def log_capture(self, caplog: pytest.LogCaptureFixture):
        logger = logging.getLogger(_DISPATCHER_LOGGER)
        logger.addHandler(caplog.handler)
        caplog.handler.setLevel(logging.DEBUG)
        yield caplog
        logger.removeHandler(caplog.handler)

    def test_should_log_and_swallow_action_exception(
        self, root, dispatcher, log_capture,
    ) -> None:
        def boom() -> None:
            raise RuntimeError("simulated paste failure")

        dispatcher.dispatch(boom)
        _pump_until(root, lambda: "simulated paste failure" in log_capture.text)

        assert "simulated paste failure" in log_capture.text

    def test_should_still_invoke_subsequent_actions_after_exception(
        self, root, dispatcher,
    ) -> None:
        ran_after = threading.Event()

        dispatcher.dispatch(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        dispatcher.dispatch(ran_after.set)
        _pump_until(root, ran_after.is_set)

        assert ran_after.is_set() is True
