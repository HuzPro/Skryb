import logging
import os
import threading
import time
from collections.abc import Callable
from typing import Protocol

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    EngineLoadFailedEvent,
    ErrorOccurredEvent,
    RecordingCappedEvent,
    StateChangedEvent,
    TranscriptionEmptyEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.downloads.download_manager import DownloadManager
from voicetotext.hotkeys.listener import HotkeyListener
from voicetotext.tray.duration_format import format_duration
from voicetotext.transcription_use_case import RecordAndTranscribeUseCase


_logger = logging.getLogger(__name__)

_STATE_LABELS = {
    AppState.LOADING: "Loading model...",
    AppState.IDLE: "Idle - ready",
    AppState.RECORDING: "Recording...",
    AppState.TRANSCRIBING: "Transcribing...",
    AppState.ERROR: "Error",
}

_QUIT_CLEANUP_TIMEOUT_SECONDS = 3.0


class TrayIcon(Protocol):
    """Minimal system-tray contract used by :class:`SystemTrayApp`.

    ``run(setup)`` blocks until ``stop()`` is called (or the event loop
    exits). ``update_state`` lets the adapter re-render for the current
    app state; ``visible`` toggles the icon's visibility.
    ``set_elapsed_provider`` swaps in a callable the adapter polls on
    a timer to keep an "elapsed since recording started" indicator
    live; ``None`` clears it.
    """

    visible: bool

    def update_state(self, state: AppState, label: str) -> None: ...
    def set_elapsed_provider(
        self, provider: Callable[[], float] | None,
    ) -> None: ...
    def notify(self, title: str, message: str) -> None: ...
    def run(self, setup: Callable[..., None]) -> None: ...
    def stop(self) -> None: ...


class Disposable(Protocol):
    def dispose(self) -> None: ...


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
        owned_disposables: tuple[Disposable, ...] = (),
    ) -> None:
        self._use_case = use_case
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._hotkey_listener = hotkey_listener
        self._tray_factory = tray_factory
        self._on_open_settings = on_open_settings
        self._download_manager = download_manager
        self._owned_disposables = owned_disposables
        self._icon: TrayIcon | None = None
        self._pending_engine_failure: str | None = None
        self._pending_engine_ready = False
        # ``_error_sticky`` keeps the icon in ERROR colour after an
        # error event even once the state machine has cycled back to
        # IDLE. The OS notification balloon can be silenced by Focus
        # Assist, hidden trays, or DND, so the icon is the only
        # guaranteed second channel. Cleared when the user activates
        # a new recording - that's the implicit acknowledgement.
        self._error_sticky = False

        self._event_bus.subscribe(StateChangedEvent, self._handle_state_changed)
        self._event_bus.subscribe(ErrorOccurredEvent, self._handle_error)
        self._event_bus.subscribe(TranscriptionFailedEvent, self._handle_error)
        self._event_bus.subscribe(
            EngineLoadFailedEvent, self._handle_engine_load_failed,
        )
        self._event_bus.subscribe(
            TranscriptionEmptyEvent, self._handle_transcription_empty,
        )
        self._event_bus.subscribe(
            RecordingCappedEvent, self._handle_recording_capped,
        )

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

    def pause_listener(self) -> None:
        self._hotkey_listener.pause()

    def resume_listener(self) -> None:
        self._hotkey_listener.resume()

    def _on_setup(self, icon: TrayIcon) -> None:
        icon.visible = True
        from voicetotext.core.startup_log import milestone

        milestone("tray-visible")
        self._replay_engine_outcome(icon)
        self._hotkey_listener.start()

    def _replay_engine_outcome(self, icon: TrayIcon) -> None:
        if self._pending_engine_failure is not None:
            self._error_sticky = True
            icon.notify("VoiceToText", self._pending_engine_failure)
            icon.update_state(AppState.ERROR, _STATE_LABELS[AppState.ERROR])
            return
        if self._pending_engine_ready:
            icon.update_state(AppState.IDLE, _STATE_LABELS[AppState.IDLE])

    def _current_state_label(self) -> str:
        return _STATE_LABELS.get(self._state_machine.state, "Unknown")

    def _handle_settings_clicked(self) -> None:
        if self._on_open_settings is None:
            return
        self._on_open_settings()

    def _handle_quit(self) -> None:
        # Schedule a hard exit FIRST, before any cleanup. Cleanup may
        # hang on cross-thread Qt operations (tray icon) or a stuck OS
        # keyboard hook; when that happens the timer kills the
        # process anyway, so the user never sees a "ghost" voicetotext
        # process after clicking Quit. Cleanup itself runs inline on
        # the GUI thread - QSystemTrayIcon mutations require it and
        # the previous daemon-thread design was silently hanging on
        # ``self._tray.hide()`` on some Windows versions, leaving the
        # join blocked indefinitely.
        threading.Timer(
            _QUIT_CLEANUP_TIMEOUT_SECONDS,
            lambda: os._exit(0),
        ).start()
        self._quit_cleanup()
        os._exit(0)

    def _quit_cleanup(self) -> None:
        steps: list[tuple[str, Callable[[], None]]] = [
            ("hotkey_listener.stop", self._hotkey_listener.stop),
            ("use_case.dispose", self._use_case.dispose),
        ]
        steps.extend(
            ("owned_disposable.dispose", item.dispose)
            for item in self._owned_disposables
        )
        steps.extend([
            ("download_manager.shutdown", self._shutdown_download_manager),
            ("icon.stop", self._stop_icon),
        ])
        for step_name, step in steps:
            try:
                step()
            except Exception:
                _logger.exception("quit cleanup step '%s' failed", step_name)

    def _shutdown_download_manager(self) -> None:
        if self._download_manager is not None:
            self._download_manager.shutdown()

    def _stop_icon(self) -> None:
        if self._icon is not None:
            self._icon.stop()

    def _handle_error(
        self, event: ErrorOccurredEvent | TranscriptionFailedEvent
    ) -> None:
        if self._icon is None:
            return
        self._error_sticky = True
        self._icon.notify("VoiceToText", event.error)
        # Force the icon to ERROR even for error paths that never
        # transitioned the state machine through ERROR (e.g.
        # start-recording failures, clipboard errors).
        self._icon.update_state(AppState.ERROR, _STATE_LABELS[AppState.ERROR])

    def _handle_recording_capped(self, event: RecordingCappedEvent) -> None:
        if self._icon is None:
            return
        self._icon.notify(
            "VoiceToText",
            f"Recording auto-stopped at "
            f"{format_duration(event.max_duration_seconds)} cap.",
        )

    def _handle_engine_load_failed(self, event: EngineLoadFailedEvent) -> None:
        self._pending_engine_failure = event.error
        self._pending_engine_ready = False
        if self._icon is None:
            return
        self._error_sticky = True
        self._icon.notify("VoiceToText", event.error)
        # Mirror ``_handle_error``: flip the icon to ERROR so the
        # broken state is visible even if the OS notification balloon
        # is suppressed by Focus Assist / DND.
        self._icon.update_state(AppState.ERROR, _STATE_LABELS[AppState.ERROR])

    def _handle_transcription_empty(self, event: TranscriptionEmptyEvent) -> None:
        if self._icon is None:
            return
        # Soft signal - the engine ran successfully, it just produced
        # no text (silent audio, unsupported language). No sticky
        # ERROR icon; just inform the user so they know why the
        # clipboard is empty.
        self._icon.notify("VoiceToText", "No speech detected.")

    def _handle_state_changed(self, event: StateChangedEvent) -> None:
        if (
            event.old_state == AppState.LOADING.value
            and event.new_state == AppState.IDLE.value
        ):
            self._pending_engine_ready = True
            self._pending_engine_failure = None
        if self._icon is None:
            return
        new_state = AppState(event.new_state)
        if new_state == AppState.RECORDING:
            self._error_sticky = False
        # The elapsed-time provider tracks the domain state, not the
        # icon presentation - install/clear it regardless of whether
        # the error sticky is suppressing the icon update.
        self._update_elapsed_tracking(
            old_state=AppState(event.old_state),
            new_state=new_state,
        )
        if self._error_sticky and new_state != AppState.ERROR:
            return
        self._icon.update_state(
            new_state, _STATE_LABELS.get(new_state, "Unknown"),
        )

    def _update_elapsed_tracking(
        self, *, old_state: AppState, new_state: AppState,
    ) -> None:
        if new_state == AppState.RECORDING:
            recording_started_at = time.monotonic()
            self._icon.set_elapsed_provider(
                lambda: time.monotonic() - recording_started_at,
            )
            return
        if old_state == AppState.RECORDING:
            self._icon.set_elapsed_provider(None)
