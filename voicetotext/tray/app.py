import os
from collections.abc import Callable
from typing import Protocol

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    ErrorOccurredEvent,
    StateChangedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


_STATE_LABELS = {
    AppState.IDLE: "Idle — ready",
    AppState.RECORDING: "Recording...",
    AppState.TRANSCRIBING: "Transcribing...",
    AppState.ERROR: "Error",
}


class TrayIcon(Protocol):
    """Minimal system-tray contract used by :class:`SystemTrayApp`.

    ``run(setup)`` blocks until ``stop()`` is called (or the event loop
    exits). ``update_state`` lets the adapter re-render for the current
    app state; ``visible`` toggles the icon's visibility.
    """

    visible: bool

    def update_state(self, state: AppState, label: str) -> None: ...
    def notify(self, title: str, message: str) -> None: ...
    def run(self, setup: Callable[..., None]) -> None: ...
    def stop(self) -> None: ...


TrayIconFactory = Callable[..., TrayIcon]


class SystemTrayApp:
    """Composition root for the tray-resident portion of the app.

    Owns the hotkey listener, state machine wiring, and the settings
    single-instance guard. Delegates all tray-library specifics to an
    injected :class:`TrayIconFactory`, so the tray backend is swappable
    without touching this class.
    """

    def __init__(
        self,
        use_case: RecordAndTranscribeUseCase,
        state_machine: AppStateMachine,
        event_bus: EventBus,
        hotkey_listener: HotkeyListener,
        tray_factory: TrayIconFactory,
        on_open_settings: Callable[[], None] | None = None,
        download_manager: DownloadManager | None = None,
    ) -> None:
        self._use_case = use_case
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._hotkey_listener = hotkey_listener
        self._tray_factory = tray_factory
        self._on_open_settings = on_open_settings
        self._download_manager = download_manager
        self._icon: TrayIcon | None = None

        self._event_bus.subscribe(StateChangedEvent, self._handle_state_changed)
        self._event_bus.subscribe(ErrorOccurredEvent, self._handle_error)
        self._event_bus.subscribe(TranscriptionFailedEvent, self._handle_error)

    def run(self) -> None:
        self._icon = self._tray_factory(
            initial_state=self._state_machine.state,
            get_state_label=self._current_state_label,
            on_settings_clicked=(
                self._handle_settings_clicked
                if self._on_open_settings is not None
                else None
            ),
            on_quit_clicked=self._handle_quit,
        )
        self._icon.run(setup=self._on_setup)

    def set_hotkey_listener(self, listener: HotkeyListener) -> None:
        self._hotkey_listener.stop()
        self._hotkey_listener = listener
        self._hotkey_listener.start()

    def _on_setup(self, icon: TrayIcon) -> None:
        icon.visible = True
        self._hotkey_listener.start()

    def _current_state_label(self) -> str:
        return _STATE_LABELS.get(self._state_machine.state, "Unknown")

    def _handle_settings_clicked(self) -> None:
        if self._on_open_settings is None:
            return
        self._on_open_settings()

    def _handle_quit(self) -> None:
        self._hotkey_listener.stop()
        self._use_case.dispose()
        if self._download_manager is not None:
            self._download_manager.shutdown()
        if self._icon is not None:
            self._icon.stop()
        os._exit(0)

    def _handle_error(
        self, event: ErrorOccurredEvent | TranscriptionFailedEvent
    ) -> None:
        if self._icon is None:
            return
        self._icon.notify("VoiceToText", event.error)

    def _handle_state_changed(self, event: StateChangedEvent) -> None:
        if self._icon is None:
            return
        new_state = AppState(event.new_state)
        self._icon.update_state(
            new_state, _STATE_LABELS.get(new_state, "Unknown")
        )
