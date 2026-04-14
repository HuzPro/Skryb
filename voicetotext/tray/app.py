import os
import threading
from collections.abc import Callable
from typing import Protocol

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import StateChangedEvent
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.tray.icons import create_icon
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


_STATE_LABELS = {
    AppState.IDLE: "Idle — ready",
    AppState.RECORDING: "Recording...",
    AppState.TRANSCRIBING: "Transcribing...",
    AppState.ERROR: "Error",
}


class TrayIcon(Protocol):
    visible: bool
    icon: object
    title: str

    def run(self, setup: Callable[..., None]) -> None: ...
    def stop(self) -> None: ...


class SystemTrayApp:

    def __init__(
        self,
        use_case: RecordAndTranscribeUseCase,
        state_machine: AppStateMachine,
        event_bus: EventBus,
        hotkey_listener: HotkeyListener,
        on_open_settings: Callable[[], None] | None = None,
        download_manager: DownloadManager | None = None,
    ) -> None:
        self._use_case = use_case
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._hotkey_listener = hotkey_listener
        self._on_open_settings = on_open_settings
        self._download_manager = download_manager
        self._icon: TrayIcon | None = None
        self._settings_lock = threading.Lock()

        self._event_bus.subscribe(StateChangedEvent, self._on_state_changed)

    def run(self) -> None:
        import pystray
        from pystray import Menu, MenuItem

        menu_items = [
            MenuItem(
                lambda item: _STATE_LABELS.get(
                    self._state_machine.state, "Unknown"
                ),
                action=None,
                enabled=False,
            ),
            Menu.SEPARATOR,
        ]

        if self._on_open_settings:
            menu_items.append(MenuItem("Settings...", self._on_settings_clicked))
            menu_items.append(Menu.SEPARATOR)

        menu_items.append(MenuItem("Quit", self._on_quit))

        self._icon = pystray.Icon(
            name="VoiceToText",
            icon=create_icon(AppState.IDLE),
            title="VoiceToText — Idle",
            menu=Menu(*menu_items),
        )
        self._icon.run(setup=self._on_setup)

    def set_hotkey_listener(self, listener: HotkeyListener) -> None:
        self._hotkey_listener.stop()
        self._hotkey_listener = listener
        self._hotkey_listener.start()

    def _on_setup(self, icon: TrayIcon) -> None:
        icon.visible = True
        self._hotkey_listener.start()

    def _on_settings_clicked(self, icon: TrayIcon, item: object) -> None:
        if self._on_open_settings:
            threading.Thread(target=self._open_settings_guarded, daemon=True).start()

    def _open_settings_guarded(self) -> None:
        if not self._settings_lock.acquire(blocking=False):
            return
        try:
            self._on_open_settings()
        finally:
            self._settings_lock.release()

    def _on_quit(self, icon: TrayIcon, item: object) -> None:
        self._hotkey_listener.stop()
        self._use_case.dispose()
        if self._download_manager is not None:
            self._download_manager.shutdown()
        icon.stop()
        os._exit(0)

    def _on_state_changed(self, event: StateChangedEvent) -> None:
        if not self._icon:
            return
        new_state = AppState(event.new_state)
        self._icon.icon = create_icon(new_state)
        self._icon.title = f"VoiceToText — {_STATE_LABELS.get(new_state, 'Unknown')}"
