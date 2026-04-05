import threading

import pytest

from voicetotext.core.event_bus import EventBus, SimpleEventBus
from voicetotext.core.events import (
    DomainEvent,
    RecordingStartedEvent,
    RecordingStoppedEvent,
    TranscriptionCompletedEvent,
)


class TestSimpleEventBusSubscription:
    def test_should_deliver_event_to_subscriber(self):
        bus = SimpleEventBus()
        received = []
        bus.subscribe(RecordingStartedEvent, received.append)

        bus.publish(RecordingStartedEvent())

        assert len(received) == 1
        assert isinstance(received[0], RecordingStartedEvent)

    def test_should_deliver_to_multiple_subscribers(self):
        bus = SimpleEventBus()
        first_received = []
        second_received = []
        bus.subscribe(RecordingStartedEvent, first_received.append)
        bus.subscribe(RecordingStartedEvent, second_received.append)

        bus.publish(RecordingStartedEvent())

        assert len(first_received) == 1
        assert len(second_received) == 1

    def test_should_not_deliver_to_unrelated_subscribers(self):
        bus = SimpleEventBus()
        received = []
        bus.subscribe(RecordingStoppedEvent, received.append)

        bus.publish(RecordingStartedEvent())

        assert len(received) == 0

    def test_should_handle_publish_with_no_subscribers(self):
        bus = SimpleEventBus()

        bus.publish(RecordingStartedEvent())


class TestSimpleEventBusUnsubscription:
    def test_should_stop_delivering_after_unsubscribe(self):
        bus = SimpleEventBus()
        received = []
        handler = received.append
        bus.subscribe(RecordingStartedEvent, handler)

        bus.unsubscribe(RecordingStartedEvent, handler)
        bus.publish(RecordingStartedEvent())

        assert len(received) == 0

    def test_should_ignore_unsubscribe_for_unknown_handler(self):
        bus = SimpleEventBus()

        bus.unsubscribe(RecordingStartedEvent, lambda e: None)


class TestSimpleEventBusThreadSafety:
    def test_should_handle_concurrent_publish_and_subscribe(self):
        bus = SimpleEventBus()
        received = []
        bus.subscribe(RecordingStartedEvent, received.append)

        errors = []

        def publish_many():
            try:
                for _ in range(100):
                    bus.publish(RecordingStartedEvent())
            except Exception as e:
                errors.append(e)

        def subscribe_many():
            try:
                for _ in range(100):
                    bus.subscribe(RecordingStoppedEvent, lambda e: None)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=publish_many),
            threading.Thread(target=subscribe_many),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(received) == 100


class TestSimpleEventBusIsAnEventBus:
    def test_should_implement_event_bus_interface(self):
        bus = SimpleEventBus()

        assert isinstance(bus, EventBus)
