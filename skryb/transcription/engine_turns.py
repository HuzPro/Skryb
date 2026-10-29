"""Whose turn it is on a speech model that dictation and a meeting share."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class EngineTurns:
    """One decode at a time, and a waiting dictation always goes before a waiting meeting clip.

    A dictation therefore waits behind at most the one meeting clip already
    decoding; meeting clips wait while any dictation is queued or decoding.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._busy = False
        self._dictations_waiting = 0

    @contextmanager
    def dictation(self) -> Iterator[None]:
        with self._changed:
            self._dictations_waiting += 1
            try:
                self._changed.wait_for(lambda: not self._busy)
            finally:
                self._dictations_waiting -= 1
            self._busy = True
        try:
            yield
        finally:
            self._release()

    @contextmanager
    def meeting(self) -> Iterator[None]:
        with self._changed:
            self._changed.wait_for(lambda: not self._busy and not self._dictations_waiting)
            self._busy = True
        try:
            yield
        finally:
            self._release()

    def _release(self) -> None:
        with self._changed:
            self._busy = False
            self._changed.notify_all()
