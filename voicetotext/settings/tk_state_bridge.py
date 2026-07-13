from collections.abc import Callable
from typing import Protocol

from voicetotext.core.events import StateChangedEvent
from voicetotext.settings.service import StateChangeListener

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]


class StateTarget(Protocol):
    def on_state_changed(self, event: StateChangedEvent) -> None: ...


class TkStateBridge(StateChangeListener):
    """Marshals app-state transitions onto the Tk GUI thread.

    Subscribed to :class:`SettingsService` as a
    :class:`StateChangeListener`. State transitions are published from
    recording/transcription worker threads, but the window's slots must
    run on the GUI thread. ``schedule`` (a thread-safe enqueue, e.g.
    :meth:`GuiThreadDispatcher.dispatch`) hands the call back to the GUI
    loop, replacing Qt's ``AutoConnection`` queueing.
    """

    def __init__(self, schedule: ScheduleOnGuiThread, target: StateTarget) -> None:
        self._schedule = schedule
        self._target = target

    def on_state_changed(self, event: StateChangedEvent) -> None:
        self._schedule(lambda: self._target.on_state_changed(event))
