from unittest.mock import MagicMock

import pytest

from tests.fakes.dispatcher import DeferredActionDispatcher, InlineActionDispatcher
from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.core.exceptions import EngineSwapError
from voicetotext.settings.engine_hot_swap import (
    EngineHotSwapCoordinator,
    NullEngineHotSwap,
)


def _collect(bus: SimpleEventBus, *event_types) -> list:
    events: list = []
    for et in event_types:
        bus.subscribe(et, events.append)
    return events


class TestEngineHotSwapCoordinatorStartSwap:
    def test_should_publish_load_started_event(self):
        bus = SimpleEventBus()
        started = _collect(bus, EngineLoadStartedEvent)

        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=lambda: MagicMock(),
            action_dispatcher=InlineActionDispatcher(),
        )
        coord.start_swap("small")

        assert len(started) == 1
        assert started[0].model_name == "small"

    def test_should_replace_engine_on_use_case_when_loader_succeeds(self):
        bus = SimpleEventBus()
        use_case = MagicMock()
        new_engine = MagicMock()

        coord = EngineHotSwapCoordinator(
            use_case=use_case,
            event_bus=bus,
            loader=lambda: new_engine,
            action_dispatcher=InlineActionDispatcher(),
        )
        coord.start_swap("small")

        use_case.set_engine.assert_called_once_with(new_engine)

    def test_should_load_new_engine_before_installing_it(self):
        bus = SimpleEventBus()
        use_case = MagicMock()
        call_order: list[str] = []

        def loader():
            call_order.append("load")
            return MagicMock()

        use_case.set_engine.side_effect = lambda _e: call_order.append("install")

        coord = EngineHotSwapCoordinator(
            use_case=use_case,
            event_bus=bus,
            loader=loader,
            action_dispatcher=InlineActionDispatcher(),
        )
        coord.start_swap("small")

        assert call_order == ["load", "install"]

    def test_should_not_call_dispose_current_engine_during_swap(self):
        use_case = MagicMock()

        coord = EngineHotSwapCoordinator(
            use_case=use_case,
            event_bus=SimpleEventBus(),
            loader=lambda: MagicMock(),
            action_dispatcher=InlineActionDispatcher(),
        )
        coord.start_swap("small")

        use_case.dispose_current_engine.assert_not_called()

    def test_should_not_dispose_new_engine_on_happy_path(self):
        new_engine = MagicMock()

        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=SimpleEventBus(),
            loader=lambda: new_engine,
            action_dispatcher=InlineActionDispatcher(),
        )
        coord.start_swap("small")

        new_engine.dispose.assert_not_called()

    def test_should_dispose_new_engine_when_set_engine_rejects(self):
        new_engine = MagicMock()
        use_case = MagicMock()
        use_case.set_engine.side_effect = EngineSwapError(
            "Can only swap engine when idle"
        )

        coord = EngineHotSwapCoordinator(
            use_case=use_case,
            event_bus=SimpleEventBus(),
            loader=lambda: new_engine,
            action_dispatcher=InlineActionDispatcher(),
        )
        coord.start_swap("small")

        new_engine.dispose.assert_called_once_with()

    def test_should_publish_failed_event_when_set_engine_rejects_non_idle_state(self):
        bus = SimpleEventBus()
        failed = _collect(bus, EngineLoadFailedEvent)
        use_case = MagicMock()
        use_case.set_engine.side_effect = EngineSwapError(
            "Can only swap engine when idle"
        )

        coord = EngineHotSwapCoordinator(
            use_case=use_case,
            event_bus=bus,
            loader=lambda: MagicMock(),
            action_dispatcher=InlineActionDispatcher(),
        )
        coord.start_swap("small")

        assert len(failed) == 1
        assert "idle" in failed[0].error.lower()

    def test_should_not_touch_use_case_engine_when_loader_fails(self):
        use_case = MagicMock()

        def broken_loader():
            raise RuntimeError("cuda oom")

        coord = EngineHotSwapCoordinator(
            use_case=use_case,
            event_bus=SimpleEventBus(),
            loader=broken_loader,
            action_dispatcher=InlineActionDispatcher(),
        )
        coord.start_swap("large-v3")

        use_case.dispose_current_engine.assert_not_called()
        use_case.set_engine.assert_not_called()

    def test_should_publish_failed_event_on_loader_exception(self):
        bus = SimpleEventBus()
        failed = _collect(bus, EngineLoadFailedEvent)

        def broken_loader():
            raise RuntimeError("cuda oom")

        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=broken_loader,
            action_dispatcher=InlineActionDispatcher(),
        )
        coord.start_swap("large-v3")

        assert len(failed) == 1
        assert failed[0].model_name == "large-v3"
        assert "cuda oom" in failed[0].error

    def test_should_raise_when_swap_already_in_progress(self):
        dispatcher = DeferredActionDispatcher()
        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=SimpleEventBus(),
            loader=lambda: MagicMock(),
            action_dispatcher=dispatcher,
        )
        coord.start_swap("small")

        with pytest.raises(EngineSwapError, match="already in progress"):
            coord.start_swap("small")

    def test_should_allow_new_swap_after_previous_finishes(self):
        bus = SimpleEventBus()
        completed = _collect(bus, EngineLoadCompletedEvent)
        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=bus,
            loader=lambda: MagicMock(),
            action_dispatcher=InlineActionDispatcher(),
        )

        coord.start_swap("tiny")
        coord.start_swap("small")

        assert len(completed) == 2

    def test_null_hot_swap_reports_not_busy(self):
        null = NullEngineHotSwap()

        assert null.is_busy() is False

    def test_null_hot_swap_raises_when_start_called(self):
        null = NullEngineHotSwap()

        with pytest.raises(EngineSwapError, match="first-run"):
            null.start_swap("base")

    def test_is_busy_reflects_swap_lifecycle(self):
        dispatcher = DeferredActionDispatcher()
        coord = EngineHotSwapCoordinator(
            use_case=MagicMock(),
            event_bus=SimpleEventBus(),
            loader=lambda: MagicMock(),
            action_dispatcher=dispatcher,
        )

        assert coord.is_busy() is False
        coord.start_swap("small")
        assert coord.is_busy() is True

        dispatcher.drain_all()
        assert coord.is_busy() is False
