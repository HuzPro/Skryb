import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable


_logger = logging.getLogger(__name__)


class ActionDispatcher(ABC):

    @abstractmethod
    def dispatch(self, action: Callable[[], None]) -> None:
        pass


class ThreadedActionDispatcher(ActionDispatcher):

    def dispatch(self, action: Callable[[], None]) -> None:
        threading.Thread(target=self._run, args=(action,), daemon=True).start()

    @staticmethod
    def _run(action: Callable[[], None]) -> None:
        # Top-level catch so an unhandled exception in a dispatched
        # action can't slip past Python's default thread excepthook (which
        # only prints to stderr — easily missed when the log file is the
        # canonical record). Native crashes still bypass this; for those
        # the only signal is the absence of further log output.
        try:
            action()
        except Exception:
            _logger.exception("dispatched action failed")
