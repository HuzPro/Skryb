from collections.abc import Callable
from typing import Protocol

from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.settings.service import EngineEventListener

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]


class EngineTarget(Protocol):
    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None: ...
    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None: ...
    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None: ...


class TkEngineBridge(EngineEventListener):
    """Marshals engine load events onto the Tk GUI thread.

    Engine load events are published from the hot-swap worker thread;
    ``schedule`` queues each onto the GUI loop before invoking the
    window's slot, replacing Qt's ``AutoConnection`` queueing.
    """

    def __init__(self, schedule: ScheduleOnGuiThread, target: EngineTarget) -> None:
        self._schedule = schedule
        self._target = target

    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None:
        self._schedule(lambda: self._target.on_engine_load_started(event))

    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        self._schedule(lambda: self._target.on_engine_load_completed(event))

    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None:
        self._schedule(lambda: self._target.on_engine_load_failed(event))
