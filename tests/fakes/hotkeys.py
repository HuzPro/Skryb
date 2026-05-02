from collections.abc import Callable

from voicetotext.hotkeys.listener import HotkeyListener


class FakeHotkeyListener(HotkeyListener):
    """In-memory HotkeyListener.

    Instead of hooking real keyboard events, tests drive activation
    through :meth:`fire_activate` and :meth:`fire_deactivate`.
    """

    def __init__(self) -> None:
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.pause_calls: int = 0
        self.resume_calls: int = 0
        self._on_activate: Callable[[], None] | None = None
        self._on_deactivate: Callable[[], None] | None = None
        self._started: bool = False

    def start(self) -> None:
        self.start_calls += 1
        self._started = True

    def stop(self) -> None:
        self.stop_calls += 1
        self._started = False

    def set_on_activate(self, callback: Callable[[], None]) -> None:
        self._on_activate = callback

    def set_on_deactivate(self, callback: Callable[[], None]) -> None:
        self._on_deactivate = callback

    def pause(self) -> None:
        self.pause_calls += 1

    def resume(self) -> None:
        self.resume_calls += 1

    def fire_activate(self) -> None:
        if not self._started:
            raise RuntimeError("fire_activate called before start()")
        if self._on_activate is None:
            raise RuntimeError("fire_activate called before set_on_activate()")
        self._on_activate()

    def fire_deactivate(self) -> None:
        if not self._started:
            raise RuntimeError("fire_deactivate called before start()")
        if self._on_deactivate is None:
            raise RuntimeError("fire_deactivate called before set_on_deactivate()")
        self._on_deactivate()
