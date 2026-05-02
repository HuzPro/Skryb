from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from voicetotext.output.gui_thread_dispatcher import GuiThreadDispatcher


class QtGuiThreadDispatcher(QObject):
    """Qt-backed :class:`GuiThreadDispatcher` using a queued signal.

    The dispatcher must be constructed on the GUI thread so its
    :class:`QObject` thread affinity points there. Calls to
    :meth:`dispatch` from any other thread emit
    ``_action_requested``; ``Qt.AutoConnection`` resolves to
    ``QueuedConnection`` because the receiver lives on a different
    thread, so the action runs on the GUI event loop. Same-thread
    calls become ``DirectConnection`` — cheap.

    Inheritance is single (``QObject`` only) because ``shiboken``'s
    metaclass conflicts with :class:`abc.ABCMeta`; we register as a
    virtual subclass via :meth:`ABC.register` at module import time —
    same workaround as :class:`QtDownloadBridge`.
    """

    _action_requested = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._action_requested.connect(self._invoke)

    def dispatch(self, action: Callable[[], None]) -> None:
        self._action_requested.emit(action)

    def _invoke(self, action: Callable[[], None]) -> None:
        action()


GuiThreadDispatcher.register(QtGuiThreadDispatcher)
