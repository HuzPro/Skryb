import threading
import time
from unittest.mock import MagicMock

import pytest

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.settings.engine_hot_swap import (
    EngineHotSwapCoordinator,
    NullEngineHotSwap,
)


class _EventCollector:
    """Records published events of a given type."""

    def __init__(self, bus: SimpleEventBus, *event_types):
        self.events: list = []
        self._bus = bus
        for et in event_types:
            bus.subscribe(et, self._on_event)

    def _on_event(self, event) -> None:
        self.events.append(event)

    def wait_until(self, predicate, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False


class TestEngineHotSwapCoordinatorStartSwap:
    def test_should_publish_load_started_event(self):
        bus = SimpleEventBus()
        use_case = MagicMock()
        collector = _EventCollector(bus, EngineLoadStartedEvent)
        loader = MagicMock(return_value=MagicMock())

        coord = EngineHotSwapCoordinator(use_case, bus, loader)
        coord.start_swap("small")

        assert collector.wait_until(lambda: len(collector.events) >= 1)
        assert collector.events[0].model_name == "small"

    def test_should_replace_engine_on_use_case_when_loader_succeeds(self):
        bus = SimpleEventBus()
        use_case = MagicMock()
        new_engine = MagicMock()
        collector = _EventCollector(bus, EngineLoadCompletedEvent)

        coord = EngineHotSwapCoordinator(
            use_case=use_case,
            event_bus=bus,
            loader=lambda: new_engine,
        )
        coord.start_swap("small")

        assert collector.wait_until(lambda: len(collector.events) >= 1)
        use_case.set_engine.assert_called_once_with(new_engine)

    def test_should_publish_failed_event_when_set_engine_rejects_non_idle_state(self):
        bus = SimpleEventBus()
        use_case = MagicMock()
        use_case.set_engine.side_effect = RuntimeError("Can only swap engine when idle")
        collector = _EventCollector(bus, EngineLoadFailedEvent)

        coord = EngineHotSwapCoordinator(
            use_case=use_case,
            event_bus=bus,
            loader=lambda: MagicMock(),
        )
        coord.start_swap("small")

        assert collector.wait_until(lambda: len(collector.events) >= 1)
        assert "idle" in collector.events[0].error.lower()

    def test_should_publish_failed_event_on_loader_exception(self):
        bus = SimpleEventBus()
        collector = _EventCollector(bus, EngineLoadFailedEvent)

        def broken_loader():
            raise RuntimeError("cuda oom")

        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=broken_loader,
        )
        coord.start_swap("large-v3")

        assert collector.wait_until(lambda: len(collector.events) >= 1)
        assert collector.events[0].model_name == "large-v3"
        assert "cuda oom" in collector.events[0].error

    def test_should_raise_when_swap_already_in_progress(self):
        release = threading.Event()

        def slow_loader():
            release.wait(timeout=2)
            return MagicMock()

        bus = SimpleEventBus()
        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=slow_loader,
        )
        coord.start_swap("small")

        try:
            with pytest.raises(RuntimeError):
                coord.start_swap("small")
        finally:
            release.set()

    def test_should_allow_new_swap_after_previous_finishes(self):
        bus = SimpleEventBus()
        collector = _EventCollector(
            bus, EngineLoadCompletedEvent, EngineLoadFailedEvent
        )
        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=lambda: MagicMock(),
        )

        coord.start_swap("tiny")
        assert collector.wait_until(lambda: len(collector.events) >= 1)

        coord.start_swap("small")
        assert collector.wait_until(lambda: len(collector.events) >= 2)

    def test_null_hot_swap_reports_not_busy(self):
        null = NullEngineHotSwap()

        assert null.is_busy() is False

    def test_null_hot_swap_raises_when_start_called(self):
        null = NullEngineHotSwap()

        with pytest.raises(RuntimeError, match="first-run"):
            null.start_swap("base")

    def test_is_busy_reflects_swap_lifecycle(self):
        release = threading.Event()

        def slow_loader():
            release.wait(timeout=2)
            return MagicMock()

        bus = SimpleEventBus()
        collector = _EventCollector(bus, EngineLoadCompletedEvent)
        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=slow_loader,
        )

        assert coord.is_busy() is False
        coord.start_swap("small")
        assert coord.is_busy() is True

        release.set()
        assert collector.wait_until(lambda: len(collector.events) >= 1)
        # Let the worker thread fully exit.
        time.sleep(0.05)
        assert coord.is_busy() is False
