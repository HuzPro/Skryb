import sys
import tkinter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, ttk

from voicetotext.core.color_scheme import ColorScheme, detect_color_scheme
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
    ModelDeletedEvent,
    StateChangedEvent,
)
from voicetotext.settings.download_progress import DownloadProgressSnapshot
from voicetotext.settings.license_consent import (
    LicenseConsentPrompt,
    ask_license_consent,
    collect_license_acceptance,
)
from voicetotext.settings.service import SettingsService
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.tk_download_bridge import TkDownloadBridge
from voicetotext.settings.tk_engine_bridge import TkEngineBridge
from voicetotext.settings.tk_hotkey_capture import CaptureBridge
from voicetotext.recording.input_devices import Microphones
from voicetotext.settings.tk_pages.audio_page import TkAudioPage
from voicetotext.settings.tk_pages.dictionary_page import TkDictionaryPage
from voicetotext.settings.tk_pages.hotkeys_page import TkHotkeysPage
from voicetotext.settings.tk_pages.output_page import TkOutputPage
from voicetotext.settings.tk_pages.transcription_page import TkTranscriptionPage
from voicetotext.settings.tk_state_bridge import TkStateBridge
from voicetotext.settings.theme import Theme, apply_theme, theme_for
from voicetotext.settings.win32_ime_composition_hook import (
    Win32ImeCompositionHook,
)
from voicetotext.settings.win32_titlebar import set_titlebar_dark

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]
ChangeSetCallback = Callable[[ChangeSet], None]
CaptureStateCallback = Callable[[bool], None]

_SECTIONS = ["Transcription", "Audio", "Hotkeys", "Output", "Dictionary"]


def _run_now(action: Callable[[], None]) -> None:
    action()


def _ignore_changes(_changes: ChangeSet) -> None:
    pass


def _ignore_capture(_recording: bool) -> None:
    pass


class TkSettingsWindow(tkinter.Toplevel):
    """Top-level Tk settings window: sidebar nav + frame-swap pages + footer.

    Depends only on :class:`SettingsService` and never imports domain types
    directly. Engine, state, and download events arrive on the event bus
    from worker threads; the ``Tk*Bridge`` listeners marshal them onto the
    GUI thread via ``schedule`` before invoking this window's slots, which
    tests may also call directly.

    Created withdrawn so it does not flash during construction; the caller
    invokes :meth:`show` to display it (also the single-instance focus path).
    """

    def __init__(
        self,
        master: tkinter.Misc,
        service: SettingsService,
        *,
        capture_bridge: CaptureBridge,
        first_run: bool = False,
        on_save: ChangeSetCallback = _ignore_changes,
        on_capture_state_changed: CaptureStateCallback = _ignore_capture,
        schedule: ScheduleOnGuiThread = _run_now,
        color_scheme: ColorScheme | None = None,
        confirm_license: LicenseConsentPrompt = ask_license_consent,
        microphones: Microphones | None = None,
    ) -> None:
        super().__init__(master)
        self._microphones = microphones or Microphones()
        self._shown_page: ttk.Frame | None = None
        # Hide immediately so the half-built, not-yet-themed window never
        # flashes on screen; ``show`` reveals it once it is dark-themed.
        self.withdraw()
        self._service = service
        self._first_run = first_run
        self._on_save = on_save
        self._on_capture_state_changed = on_capture_state_changed
        self._schedule = schedule
        self._capture_bridge = capture_bridge
        self._confirm_license = confirm_license
        self._pending_changes: ChangeSet | None = None
        self._closing = False
        # ``None`` means follow the OS light/dark setting at open time.
        if color_scheme is None:
            color_scheme = detect_color_scheme()
        self._theme: Theme = theme_for(color_scheme)

        self.title("VoiceToText Setup" if first_run else "VoiceToText Settings")
        self.geometry("980x800")
        self.minsize(860, 640)
        self.protocol("WM_DELETE_WINDOW", self.close)

        self._engine_bridge = TkEngineBridge(schedule, self)
        self._state_bridge = TkStateBridge(schedule, self)
        self._download_bridge = TkDownloadBridge(schedule, self)

        self._build_ui()
        apply_theme(self, self._theme)
        set_titlebar_dark(self, dark=self._theme.is_dark)
        if sys.platform == "win32":
            self._install_ime_composition_hook()

        self._service.subscribe_engine(self._engine_bridge)
        self._service.subscribe_state_changes(self._state_bridge)
        self._service.subscribe_downloads(self._download_bridge)
        self._service.subscribe_model_deleted(
            self._download_bridge.on_model_deleted,
        )
        self._refresh_save_enabled()

    def _install_ime_composition_hook(self) -> None:
        """Best-effort: the settings window still opens without the hook."""
        try:
            self._ime_composition_hook = Win32ImeCompositionHook(self)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_footer()

        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)

        self._sidebar = tkinter.Listbox(
            body, width=16, exportselection=False, activestyle="none",
        )
        for label in _SECTIONS:
            self._sidebar.insert("end", f"  {label}")
        self._sidebar.bind("<<ListboxSelect>>", self._on_sidebar_select)
        # Inset from the window edge so the tabs are not flush against it.
        self._sidebar.pack(side="left", fill="y", padx=(12, 4), pady=(12, 0))

        content = ttk.Frame(body)
        content.pack(side="left", fill="both", expand=True, padx=(4, 12), pady=(12, 0))
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)

        draft = self._service.current_draft()
        self._transcription_page = TkTranscriptionPage(
            content, self._service, confirm_license=self._confirm_license,
        )
        self._transcription_page.download_error.connect(self._on_download_error)
        self._transcription_page.delete_error.connect(self._on_delete_error)
        self._transcription_page.model_auto_switched.connect(
            self._on_model_auto_switched,
        )
        self._hotkeys_page = TkHotkeysPage(
            content, self._capture_bridge, self._schedule,
        )
        self._hotkeys_page.recording_state_changed.connect(
            self._on_recording_state_changed,
        )
        self._audio_page = TkAudioPage(content, self._microphones)
        self._output_page = TkOutputPage(
            content, on_reset_pill_position=self._service.reset_pill_position,
        )
        self._dictionary_page = TkDictionaryPage(content)
        self._dictionary_page.set_dictionary(self._service.dictionary_rules())

        self._pages = [
            self._transcription_page,
            self._audio_page,
            self._hotkeys_page,
            self._output_page,
            self._dictionary_page,
        ]
        for page in self._pages:
            page.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

        self._apply_initial_draft(draft)

        self._sidebar.selection_set(0)
        self._show_page(0)

    def _build_footer(self) -> None:
        footer = ttk.Frame(self)
        footer.pack(side="bottom", fill="x", padx=12, pady=8)

        self._status_label = ttk.Label(
            footer, text="", foreground=self._theme.status_neutral,
        )
        self._status_label.pack(side="left", fill="x", expand=True)

        button_text = "Save & Start" if self._first_run else "Save & Apply"
        self._save_button = ttk.Button(
            footer, text=button_text, command=self._on_save_clicked,
        )
        self._save_button.pack(side="right")

        self._export_settings_button = ttk.Button(
            footer,
            text="Export...",
            command=self._on_export_settings,
        )
        self._export_settings_button.pack(side="right", padx=(0, 8))

        self._import_settings_button = ttk.Button(
            footer,
            text="Import...",
            command=self._on_import_settings,
        )
        self._import_settings_button.pack(side="right", padx=(0, 8))

        self._logs_folder_button = ttk.Button(
            footer,
            text="Logs folder...",
            command=self._on_choose_logs_folder,
        )
        self._logs_folder_button.pack(side="right", padx=(0, 8))

    def _apply_initial_draft(self, draft: SettingsDraft) -> None:
        self._hotkeys_page.set_combination(draft.hotkey_combination)
        self._hotkeys_page.set_mode(draft.hotkey_mode)
        self._output_page.set_mode(draft.output_mode)
        self._output_page.set_capture_content(draft.capture_content)
        self._audio_page.set_device(draft.input_device)
        self._audio_page.set_boost_quiet_speech(draft.boost_quiet_speech)
        self._audio_page.set_recording_preferences(draft.recording)

    def _on_choose_logs_folder(self) -> None:
        selected = filedialog.askdirectory(
            parent=self,
            title="Choose where session logs are saved",
            initialdir=self._service.diagnostics_directory() or None,
            mustexist=False,
        )
        if not selected:
            return
        try:
            self._service.set_diagnostics_directory(selected)
        except Exception as exc:  # noqa: BLE001
            self._show_status(f"Could not set logs folder: {exc}", error=True)
            return
        self._show_status("Logs folder applies from the next launch.")

    def _on_export_settings(self) -> None:
        selected = filedialog.asksaveasfilename(
            parent=self,
            title="Export settings",
            defaultextension=".json",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return
        try:
            self._service.export_settings(Path(selected))
        except Exception as exc:  # noqa: BLE001
            self._show_status(f"Could not export settings: {exc}", error=True)
            return
        self._show_status("Settings exported.")

    def _on_import_settings(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self,
            title="Import settings",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return
        try:
            self._service.import_settings(Path(selected))
        except Exception as exc:  # noqa: BLE001
            self._show_status(f"Could not import settings: {exc}", error=True)
            return
        self._apply_imported_draft()
        self._show_status("Settings imported.")

    def _apply_imported_draft(self) -> None:
        draft = self._service.current_draft()
        self._transcription_page.apply_mode(draft.transcription_mode)
        self._transcription_page.set_model(draft.model_size)
        self._transcription_page.set_language(draft.language)
        self._transcription_page.set_engine(draft.engine)
        self._transcription_page.set_crisperwhisper_mode(draft.crisperwhisper_mode)
        self._apply_initial_draft(draft)

    def _on_sidebar_select(self, _event: tkinter.Event) -> None:
        selection = self._sidebar.curselection()
        if selection:
            self._show_page(selection[0])

    def _show_page(self, index: int) -> None:
        page = self._pages[index]
        if self._shown_page is self._audio_page and page is not self._audio_page:
            self._audio_page.on_hidden()
        page.tkraise()
        if page is self._audio_page and self._shown_page is not page:
            self._audio_page.on_shown()
        self._shown_page = page

    def show(self) -> None:
        """Reveal and focus the window (also the single-instance focus path)."""
        # Set the dark caption BEFORE mapping: the title bar is painted when
        # the window maps, so the attribute must already be in place.
        set_titlebar_dark(self, dark=self._theme.is_dark)
        self.deiconify()
        self.lift()
        self.focus_force()

    def is_open(self) -> bool:
        """True while the window is alive and not in the process of closing."""
        return bool(self.winfo_exists()) and not self._closing

    # ------------------------------------------------------------------
    # Save flow
    # ------------------------------------------------------------------

    def collect_draft(self) -> SettingsDraft:
        return SettingsDraft(
            model_size=self._transcription_page.current_model(),
            language=self._transcription_page.current_language(),
            device=self._service.current_draft().device,
            engine=self._transcription_page.current_engine(),
            hotkey_combination=self._hotkeys_page.current_combination(),
            hotkey_mode=self._hotkeys_page.current_mode(),
            output_mode=self._output_page.current_mode(),
            capture_content=self._output_page.captures_content(),
            input_device=self._audio_page.current_device(),
            boost_quiet_speech=self._audio_page.boosts_quiet_speech(),
            recording=self._audio_page.recording_preferences(),
            transcription_mode=self._transcription_page.current_mode(),
            crisperwhisper_mode=(
                self._transcription_page.current_crisperwhisper_mode()
            ),
        )

    def _on_save_clicked(self) -> None:
        if self._first_run and not self._service.any_model_installed():
            self._show_status(
                "Please download at least one model before saving.",
                error=True,
            )
            return

        draft = self.collect_draft()
        if not self._service.is_installed(draft.model_size):
            self._show_status(
                f"'{draft.model_size}' is not installed. Download it first.",
                error=True,
            )
            return
        if not self._license_accepted_for(draft.model_size):
            self._show_status(
                f"'{draft.model_size}' needs its licence accepted before it "
                "can be used. Nothing was changed.",
                error=True,
            )
            return
        changes = self._service.apply_draft(draft)
        self._pending_changes = self._save_dictionary_edits(changes)

        if changes.engine_reload_required and not self._first_run:
            self._disable_save_button()
            self._show_status("Loading model, please wait...")
            self._service.trigger_engine_swap()
            return

        self._finalize_save()

    def _license_accepted_for(self, model_name: str) -> bool:
        """Collect consent before Save commits anything.

        Save both persists the draft and closes this window, so a licence
        asked for afterwards would have no surface left to accept on.
        """
        return collect_license_acceptance(
            self._service, model_name, self, self._confirm_license,
        )

    def _save_dictionary_edits(self, changes: ChangeSet) -> ChangeSet:
        """Persist staged dictionary rules and flag the output chain for rebuild.

        The rules live in the outermost output decorator, so rebuilding
        output is exactly what makes new rules take effect without a
        restart.
        """
        if not self._dictionary_page.is_dirty():
            return changes
        self._service.save_dictionary_rules(self._dictionary_page.dictionary())
        return replace(changes, output_changed=True)

    def _finalize_save(self) -> None:
        changes = self._pending_changes
        self._pending_changes = None
        if changes is not None:
            self._on_save(changes)
        self.close()

    def _disable_save_button(self) -> None:
        self._save_button.state(["disabled"])

    def _enable_save_button(self) -> None:
        self._save_button.state(["!disabled"])

    def _refresh_save_enabled(self) -> None:
        allowed = self._service.can_apply_now() and self._pending_changes is None
        if allowed:
            self._enable_save_button()
        else:
            self._disable_save_button()

    def _show_status(self, message: str, error: bool = False) -> None:
        self._status_label.configure(
            text=message,
            foreground=(
                self._theme.status_error if error else self._theme.status_neutral
            ),
        )

    # ------------------------------------------------------------------
    # Engine slots
    # ------------------------------------------------------------------

    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None:
        self._show_status(f"Loading '{event.model_name}'...")

    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        self._enable_save_button()
        self._show_status(f"Loaded '{event.model_name}'.")
        self._finalize_save()
        self._refresh_save_enabled()

    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None:
        self._enable_save_button()
        self._pending_changes = None
        self._show_status(
            f"Failed to load '{event.model_name}': {event.error}", error=True,
        )
        self._refresh_save_enabled()

    # ------------------------------------------------------------------
    # Download slots
    # ------------------------------------------------------------------

    def on_download_progress(self, event: DownloadProgressEvent) -> None:
        snapshot = DownloadProgressSnapshot(
            state="downloading",
            bytes_done=event.bytes_done,
            bytes_total=event.bytes_total,
            speed_bps=event.speed_bps,
            eta_seconds=event.eta_seconds,
            error_message=None,
        )
        self._model_list().set_download_progress(event.job_id, snapshot)

    def on_download_state(self, event: DownloadStateChangedEvent) -> None:
        model_list = self._model_list()
        existing = model_list.download_snapshot(event.job_id)
        bytes_done = existing.bytes_done if existing else 0
        bytes_total = existing.bytes_total if existing else 0
        snapshot = DownloadProgressSnapshot(
            state=event.state,
            bytes_done=bytes_done,
            bytes_total=bytes_total,
            speed_bps=0.0,
            eta_seconds=0.0,
            error_message=None,
        )
        model_list.set_download_progress(event.job_id, snapshot)

    def on_download_completed(self, event: DownloadCompletedEvent) -> None:
        model_list = self._model_list()
        model_list.clear_download(event.job_id)
        model_list.update_install_state(event.job_id)

    def on_download_failed(self, event: DownloadFailedEvent) -> None:
        snapshot = DownloadProgressSnapshot(
            state="failed",
            bytes_done=0,
            bytes_total=0,
            speed_bps=0.0,
            eta_seconds=0.0,
            error_message=event.error,
        )
        self._model_list().set_download_progress(event.job_id, snapshot)
        self._show_status(
            f"Download '{event.job_id}' failed: {event.error}", error=True,
        )

    def on_model_deleted(self, event: ModelDeletedEvent) -> None:
        model_list = self._model_list()
        model_list.clear_download(event.name)
        model_list.update_install_state(event.name)

    def _on_download_error(self, name: str, error: str) -> None:
        self._show_status(
            f"Could not start download of '{name}': {error}", error=True,
        )

    def _on_delete_error(self, name: str, error: str) -> None:
        self._show_status(f"Cannot delete '{name}': {error}", error=True)

    def _on_model_auto_switched(self, replaced: str, new_model: str) -> None:
        self._show_status(
            f"Model switched to '{new_model}' because '{replaced}' does "
            "not support this mode.",
        )

    # ------------------------------------------------------------------
    # State-machine + capture listeners
    # ------------------------------------------------------------------

    def on_state_changed(self, _event: StateChangedEvent) -> None:
        self._refresh_save_enabled()

    def _on_recording_state_changed(self, recording: bool) -> None:
        self._on_capture_state_changed(recording)

    # ------------------------------------------------------------------
    # Close / teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._service.unsubscribe_engine(self._engine_bridge)
        self._service.unsubscribe_state_changes(self._state_bridge)
        self._service.unsubscribe_downloads(self._download_bridge)
        self._service.unsubscribe_model_deleted(
            self._download_bridge.on_model_deleted,
        )
        # Defer the actual teardown (like Qt's deleteLater) so a slot that
        # closes the window mid-callback does not destroy widgets it is
        # still executing within.
        self.after_idle(self._destroy_if_exists)

    def _destroy_if_exists(self) -> None:
        if self.winfo_exists():
            self.destroy()

    # ------------------------------------------------------------------
    # Test accessors
    # ------------------------------------------------------------------

    def transcription_page(self) -> TkTranscriptionPage:
        return self._transcription_page

    def hotkeys_page(self) -> TkHotkeysPage:
        return self._hotkeys_page

    def output_page(self) -> TkOutputPage:
        return self._output_page

    def dictionary_page(self) -> TkDictionaryPage:
        return self._dictionary_page

    def save_button(self) -> ttk.Button:
        return self._save_button

    def export_settings_button(self) -> ttk.Button:
        return self._export_settings_button

    def import_settings_button(self) -> ttk.Button:
        return self._import_settings_button

    def status_label(self) -> ttk.Label:
        return self._status_label

    def save_enabled(self) -> bool:
        return not self._save_button.instate(["disabled"])

    def sidebar_labels(self) -> list[str]:
        return [
            self._sidebar.get(i).strip() for i in range(self._sidebar.size())
        ]

    def _model_list(self):
        return self._transcription_page.model_list()
