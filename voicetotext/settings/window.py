import sys
import tkinter
from collections.abc import Callable
from tkinter import messagebox

import customtkinter as ctk

from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.settings.hotkeys_tab import HotkeyTab
from voicetotext.settings.output_tab import OutputTab
from voicetotext.settings.service import EngineEventListener, SettingsService
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.transcription_tab import TranscriptionTab


_DEFAULT_WINDOW_SIZE = "580x720"
_MIN_WINDOW_SIZE = (540, 520)
_MAX_WINDOW_SIZE = (900, 1100)


class SettingsWindow(EngineEventListener):
    """Top-level settings window. Composes the three tabs and the save flow.

    The window depends only on :class:`SettingsService` — it never imports
    domain types directly, which is what lets the UI stay testable in
    isolation from transcription, downloads, and config internals.
    """

    def __init__(
        self,
        service: SettingsService,
        first_run: bool = False,
        on_save: Callable[[ChangeSet], None] | None = None,
    ) -> None:
        self._service = service
        self._first_run = first_run
        self._on_save = on_save
        self._window: ctk.CTk | None = None
        self._status_label: ctk.CTkLabel | None = None
        self._save_button: ctk.CTkButton | None = None
        self._transcription_tab: TranscriptionTab | None = None
        self._hotkey_tab: HotkeyTab | None = None
        self._output_tab: OutputTab | None = None
        self._pending_changes: ChangeSet | None = None

    def run(self) -> None:
        # Settings is opened from a background daemon thread on every click
        # (see SystemTrayApp._on_settings_clicked). Each time we land here we
        # have a fresh thread, but ``tkinter._default_root`` may still point
        # to the ``Tk`` instance from a previous open on a now-dead thread —
        # font creation inside customtkinter walks that stale root and raises
        # ``RuntimeError: main thread is not in main loop``. Resetting the
        # module-global forces ``ctk.CTk()`` below to become the new default
        # root for *this* thread.
        tkinter._default_root = None

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self._window = ctk.CTk()
        title = "VoiceToText Setup" if self._first_run else "VoiceToText Settings"
        self._window.title(title)
        self._window.geometry(_DEFAULT_WINDOW_SIZE)
        self._window.minsize(*_MIN_WINDOW_SIZE)
        self._window.maxsize(*_MAX_WINDOW_SIZE)
        self._window.resizable(True, True)

        if self._first_run:
            self._window.protocol("WM_DELETE_WINDOW", self._on_close_first_run)
        else:
            self._window.protocol("WM_DELETE_WINDOW", self._on_close)

        self._service.subscribe_engine(self)

        self._build_ui()
        try:
            self._window.mainloop()
        finally:
            self._dispose_tabs()
            self._service.unsubscribe_engine(self)
            self._transcription_tab = None
            self._hotkey_tab = None
            self._output_tab = None
            self._window = None
            tkinter._default_root = None

    def _build_ui(self) -> None:
        if self._first_run:
            banner = ctk.CTkLabel(
                self._window,
                text="Welcome! Download a model to get started.\n'base' is recommended.",
                font=ctk.CTkFont(size=14),
                wraplength=500,
            )
            banner.pack(pady=(18, 6), padx=24)

        tabview = ctk.CTkTabview(self._window)
        tabview.pack(fill="both", expand=True, padx=18, pady=(8, 10))

        tab_transcription = tabview.add("Transcription")
        tab_hotkeys = tabview.add("Hotkeys")
        tab_output = tabview.add("Output")

        self._transcription_tab = TranscriptionTab(
            parent=tab_transcription, service=self._service
        )
        self._hotkey_tab = HotkeyTab(tab_hotkeys, self._service)
        self._output_tab = OutputTab(tab_output, self._service)

        self._status_label = ctk.CTkLabel(
            self._window,
            text="",
            text_color="#CCCCCC",
            font=ctk.CTkFont(size=12),
        )
        self._status_label.pack(pady=(0, 4), padx=18)

        button_text = "Save & Start" if self._first_run else "Save & Apply"
        self._save_button = ctk.CTkButton(
            self._window,
            text=button_text,
            command=self._on_save_clicked,
            height=38,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self._save_button.pack(pady=(0, 18), padx=18)

    # ------------------------------------------------------------------
    # Save flow
    # ------------------------------------------------------------------

    def _collect_draft(self) -> SettingsDraft:
        return SettingsDraft(
            model_size=self._transcription_tab.current_model(),
            language=self._transcription_tab.current_language(),
            device=self._transcription_tab.current_device(),
            hotkey_combination=self._hotkey_tab.current_combination(),
            hotkey_mode=self._hotkey_tab.current_mode(),
            output_mode=self._output_tab.current_mode(),
        )

    def _on_save_clicked(self) -> None:
        draft = self._collect_draft()
        changes = self._service.apply_draft(draft)

        if self._first_run and not self._service.any_model_installed():
            self._show_status(
                "Please download at least one model before starting.",
                color="#FF6666",
            )
            return

        self._pending_changes = changes

        # First-run mode has no engine to swap — build_app will load the
        # first engine fresh after the window closes.
        if changes.engine_reload_required and not self._first_run:
            # Hot-swap runs on a background thread; we wait for the
            # completion event before closing the window so the user
            # sees the loading state.
            self._disable_save_button()
            self._show_status("Loading model — please wait…", color="#CCCCCC")
            self._service.trigger_engine_swap()
            return

        self._finalize_save()

    def _finalize_save(self) -> None:
        changes = self._pending_changes
        self._pending_changes = None
        if self._on_save is not None and changes is not None:
            self._on_save(changes)
        self._window.destroy()

    def _disable_save_button(self) -> None:
        if self._save_button is not None:
            self._save_button.configure(state="disabled")

    def _enable_save_button(self) -> None:
        if self._save_button is not None:
            self._save_button.configure(state="normal")

    def _show_status(self, message: str, color: str = "#CCCCCC") -> None:
        if self._status_label is not None:
            self._status_label.configure(text=message, text_color=color)

    # ------------------------------------------------------------------
    # EngineEventListener — called from background threads
    # ------------------------------------------------------------------

    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None:
        if self._window is not None:
            self._window.after(
                0,
                self._show_status,
                f"Loading '{event.model_name}'…",
                "#CCCCCC",
            )

    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        if self._window is not None:
            self._window.after(0, self._after_engine_loaded)

    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None:
        if self._window is not None:
            self._window.after(
                0,
                self._after_engine_load_failed,
                event.model_name,
                event.error,
            )

    def _after_engine_loaded(self) -> None:
        self._enable_save_button()
        self._finalize_save()

    def _after_engine_load_failed(self, model_name: str, error: str) -> None:
        self._enable_save_button()
        self._show_status(f"Failed to load '{model_name}': {error}", color="#FF6666")
        self._pending_changes = None

    # ------------------------------------------------------------------
    # Close handlers
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        self._window.destroy()

    def _on_close_first_run(self) -> None:
        if self._service.any_model_installed():
            draft = self._collect_draft()
            self._service.apply_draft(draft)
            self._window.destroy()
        else:
            if messagebox.askyesno(
                "Quit?",
                "No model downloaded. The app needs at least one model to work.\n\nQuit anyway?",
            ):
                self._window.destroy()
                sys.exit(0)

    def _dispose_tabs(self) -> None:
        """Tear down tabs and release tkinter Variable references.

        Called in the ``finally`` block after ``mainloop()`` returns.
        Clearing the tabs' StringVar ``_tk`` references here prevents
        ``Variable.__del__`` from raising "main thread is not in main
        loop" when GC eventually collects them on the dying daemon
        thread (see ``tk_cleanup.release_tk_variables`` for details).
        """
        for tab in (
            self._transcription_tab,
            self._hotkey_tab,
            self._output_tab,
        ):
            if tab is not None:
                try:
                    tab.dispose()
                except Exception:
                    # Dispose is best-effort — swallow rather than
                    # leaking a second exception out of the finally.
                    pass
