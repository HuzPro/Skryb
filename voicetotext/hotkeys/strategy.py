from abc import ABC, abstractmethod
from collections.abc import Callable

from pynput.keyboard import Key, KeyCode


class HotkeyStrategy(ABC):

    @abstractmethod
    def on_key_press(self, key: Key | KeyCode) -> None:
        pass

    @abstractmethod
    def on_key_release(self, key: Key | KeyCode) -> None:
        pass

    @abstractmethod
    def set_callbacks(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        pass
