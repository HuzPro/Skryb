import threading
from abc import ABC, abstractmethod
from collections.abc import Callable


class ActionDispatcher(ABC):

    @abstractmethod
    def dispatch(self, action: Callable[[], None]) -> None:
        pass


class ThreadedActionDispatcher(ActionDispatcher):

    def dispatch(self, action: Callable[[], None]) -> None:
        threading.Thread(target=action, daemon=True).start()
