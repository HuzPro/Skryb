from collections.abc import Callable


class Signal:
    """A minimal synchronous Observer: a list of subscriber callbacks.

    Replaces Qt's ``Signal``/``connect``/``emit`` for the Tk view layer so
    pages and rows can publish UI events to their enclosing container
    without a toolkit dependency. Subscribers are invoked synchronously,
    in connection order, on the calling thread.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[..., None]] = []

    def connect(self, subscriber: Callable[..., None]) -> None:
        self._subscribers.append(subscriber)

    def disconnect(self, subscriber: Callable[..., None]) -> None:
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def emit(self, *args: object) -> None:
        for subscriber in list(self._subscribers):
            subscriber(*args)
