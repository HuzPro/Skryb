from skryb.core.event_bus import SimpleEventBus
from skryb.core.events import DomainEvent


class RecordingEventBus(SimpleEventBus):
    """A :class:`SimpleEventBus` that also records every published event.

    Tests that care about the overall sequence of events can read
    :attr:`history` directly rather than subscribing to each event type
    individually.  The underlying subscribe/publish behaviour is
    unchanged, so the fake is interchangeable with the real bus.
    """

    def __init__(self) -> None:
        super().__init__()
        self.history: list[DomainEvent] = []

    def publish(self, event: DomainEvent) -> None:
        self.history.append(event)
        super().publish(event)

    def events_of(self, event_type: type[DomainEvent]) -> list[DomainEvent]:
        return [e for e in self.history if isinstance(e, event_type)]
