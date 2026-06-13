from abc import ABC, abstractmethod
from collections.abc import Callable

from voicetotext.hotkeys.keys import HotkeyKey


class HotkeyStrategy(ABC):

    @abstractmethod
    def on_key_press(self, key: HotkeyKey) -> None:
        pass

    @abstractmethod
    def on_key_release(self, key: HotkeyKey) -> None:
        pass

    @abstractmethod
    def set_callbacks(
        self,
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
    ) -> None:
        pass
