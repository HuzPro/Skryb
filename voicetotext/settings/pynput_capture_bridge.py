from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Signal
from pynput import keyboard

from voicetotext.hotkeys.key_to_name import key_to_canonical_name


ListenerFactory = Callable[..., Any]


def _default_listener_factory(**kwargs: Any) -> Any:
    return keyboard.Listener(**kwargs)


class PynputCaptureBridge(QObject):

    key_pressed = Signal(str)

    def __init__(
        self,
        listener_factory: ListenerFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._listener_factory = listener_factory or _default_listener_factory
        self._listener: Any = None

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = self._listener_factory(
            on_press=self._on_press,
            suppress=True,
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None

    def _on_press(self, key: object) -> None:
        name = key_to_canonical_name(key)  # type: ignore[arg-type]
        if name is None:
            return
        self.key_pressed.emit(name)
