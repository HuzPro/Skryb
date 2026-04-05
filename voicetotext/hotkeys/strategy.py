from abc import ABC, abstractmethod
from typing import Any, Callable


class HotkeyStrategy(ABC):

    @abstractmethod
    def on_key_press(self, key: Any) -> None:
        pass

    @abstractmethod
    def on_key_release(self, key: Any) -> None:
        pass

    @abstractmethod
    def set_callbacks(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        pass
