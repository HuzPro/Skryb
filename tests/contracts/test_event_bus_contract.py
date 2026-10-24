"""Port contract tests for :class:`EventBus`.

The same suite runs against every implementation so a new subclass
(the test :class:`RecordingEventBus`, or a future in-process async
bus) can't quietly break subscribe/publish/unsubscribe semantics.
"""

from collections.abc import Callable

import pytest

from tests.fakes import RecordingEventBus
from skryb.core.event_bus import EventBus, SimpleEventBus
from skryb.core.events import DomainEvent, RecordingStartedEvent, RecordingStoppedEvent


Factory = Callable[[], EventBus]


@pytest.fixture(
    params=[
        pytest.param(SimpleEventBus, id="SimpleEventBus"),
        pytest.param(RecordingEventBus, id="RecordingEventBus"),
    ],
)
def bus(request: pytest.FixtureRequest) -> EventBus:
    factory: Factory = request.param
    return factory()


class TestEventBusContract:
    def test_should_deliver_event_to_subscribed_handler(self, bus: EventBus):
        received: list[DomainEvent] = []
        bus.subscribe(RecordingStartedEvent, received.append)

        bus.publish(RecordingStartedEvent())

        assert len(received) == 1
        assert isinstance(received[0], RecordingStartedEvent)

    def test_should_deliver_only_events_of_the_subscribed_type(
        self, bus: EventBus
    ):
        started: list[DomainEvent] = []
        bus.subscribe(RecordingStartedEvent, started.append)

        bus.publish(RecordingStoppedEvent())

        assert started == []

    def test_should_allow_multiple_handlers_for_the_same_event_type(
        self, bus: EventBus
    ):
        first: list[DomainEvent] = []
        second: list[DomainEvent] = []
        bus.subscribe(RecordingStartedEvent, first.append)
        bus.subscribe(RecordingStartedEvent, second.append)

        bus.publish(RecordingStartedEvent())

        assert len(first) == 1
        assert len(second) == 1

    def test_should_stop_delivering_events_after_unsubscribe(
        self, bus: EventBus
    ):
        received: list[DomainEvent] = []
        bus.subscribe(RecordingStartedEvent, received.append)
        bus.unsubscribe(RecordingStartedEvent, received.append)

        bus.publish(RecordingStartedEvent())

        assert received == []

    def test_should_not_raise_when_unsubscribing_unknown_handler(
        self, bus: EventBus
    ):
        def noop(event: DomainEvent) -> None: ...

        bus.unsubscribe(RecordingStartedEvent, noop)  # should not raise

    def test_should_not_raise_when_publishing_with_no_subscribers(
        self, bus: EventBus
    ):
        bus.publish(RecordingStartedEvent())  # should not raise

    def test_should_preserve_event_identity_from_publish_to_handler(
        self, bus: EventBus
    ):
        received: list[DomainEvent] = []
        bus.subscribe(RecordingStartedEvent, received.append)
        event = RecordingStartedEvent()

        bus.publish(event)

        assert received[0] is event


class TestRecordingEventBusAddsHistory:
    """RecordingEventBus extends the contract with an ordered history."""

    def test_should_record_every_published_event_in_order(self):
        bus = RecordingEventBus()

        bus.publish(RecordingStartedEvent())
        bus.publish(RecordingStoppedEvent())

        history_types = [type(e) for e in bus.history]
        assert history_types == [RecordingStartedEvent, RecordingStoppedEvent]

    def test_should_filter_history_by_event_type(self):
        bus = RecordingEventBus()

        bus.publish(RecordingStartedEvent())
        bus.publish(RecordingStoppedEvent())
        bus.publish(RecordingStartedEvent())

        started = bus.events_of(RecordingStartedEvent)
        assert len(started) == 2

    def test_should_still_deliver_events_to_subscribers(self):
        bus = RecordingEventBus()
        received: list[DomainEvent] = []
        bus.subscribe(RecordingStartedEvent, received.append)

        bus.publish(RecordingStartedEvent())

        assert len(received) == 1
