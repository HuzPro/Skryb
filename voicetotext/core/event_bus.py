import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Type

from voicetotext.core.events import DomainEvent


class EventBus(ABC):

    @abstractmethod
    def subscribe(
        self,
        event_type: Type[DomainEvent],
        handler: Callable[[DomainEvent], None],
    ) -> None:
        pass

    @abstractmethod
    def unsubscribe(
        self,
        event_type: Type[DomainEvent],
        handler: Callable[[DomainEvent], None],
    ) -> None:
        pass

    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        pass


class SimpleEventBus(EventBus):

    def __init__(self):
        self._handlers: dict[Type[DomainEvent], list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(
        self,
        event_type: Type[DomainEvent],
        handler: Callable[[DomainEvent], None],
    ) -> None:
        with self._lock:
            self._handlers[event_type].append(handler)

    def unsubscribe(
        self,
        event_type: Type[DomainEvent],
        handler: Callable[[DomainEvent], None],
    ) -> None:
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: DomainEvent) -> None:
        with self._lock:
            handlers = list(self._handlers.get(type(event), []))
        for handler in handlers:
            handler(event)
