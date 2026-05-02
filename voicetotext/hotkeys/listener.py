from abc import ABC, abstractmethod
from collections.abc import Callable


class HotkeyListener(ABC):

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def set_on_activate(self, callback: Callable[[], None]) -> None:
        pass

    @abstractmethod
    def set_on_deactivate(self, callback: Callable[[], None]) -> None:
        pass

    @abstractmethod
    def pause(self) -> None:
        pass

    @abstractmethod
    def resume(self) -> None:
        pass
