from PySide6.QtCore import QObject, Signal

from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.settings.service import EngineEventListener


class QtEngineBridge(QObject):
    """Observer that re-emits engine load events as Qt signals."""

    load_started = Signal(object)
    load_completed = Signal(object)
    load_failed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None:
        self.load_started.emit(event)

    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        self.load_completed.emit(event)

    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None:
        self.load_failed.emit(event)


EngineEventListener.register(QtEngineBridge)
