from PySide6.QtCore import QObject, Signal

from voicetotext.core.events import StateChangedEvent
from voicetotext.settings.service import StateChangeListener


class QtStateBridge(QObject):
    """Observer that re-emits app-state transitions as a Qt signal.

    Subscribed to :class:`SettingsService` as a
    :class:`StateChangeListener`. Qt's AutoConnection marshals the
    emitted signal onto the GUI thread when the receiver is a QObject
    in that thread, so slots stay thread-safe even though state
    transitions are published from recording/transcription worker
    threads.

    Inheritance is single (``QObject`` only) because ``shiboken``'s
    metaclass conflicts with :class:`abc.ABCMeta`; we register as a
    virtual subclass via :meth:`ABC.register` at module import time.
    """

    state_changed = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)

    def on_state_changed(self, event: StateChangedEvent) -> None:
        self.state_changed.emit(event)


StateChangeListener.register(QtStateBridge)
