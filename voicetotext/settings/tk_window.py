import tkinter
from collections.abc import Callable
from tkinter import ttk

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
from voicetotext.settings.service import SettingsService
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.tk_download_bridge import TkDownloadBridge
from voicetotext.settings.tk_engine_bridge import TkEngineBridge
from voicetotext.settings.tk_hotkey_capture import CaptureBridge
from voicetotext.settings.tk_pages.hotkeys_page import TkHotkeysPage
from voicetotext.settings.tk_pages.output_page import TkOutputPage
from voicetotext.settings.tk_pages.transcription_page import TkTranscriptionPage
from voicetotext.settings.tk_state_bridge import TkStateBridge

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]
ChangeSetCallback = Callable[[ChangeSet], None]
CaptureStateCallback = Callable[[bool], None]

_SECTIONS = ["Transcription", "Hotkeys", "Output"]
_STATUS_ERROR_COLOR = "#d33b3b"
_STATUS_NEUTRAL_COLOR = "#333333"


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
    ) -> None:
        super().__init__(master)
        self._service = service
        self._first_run = first_run
        self._on_save = on_save
        self._on_capture_state_changed = on_capture_state_changed
        self._schedule = schedule
        self._capture_bridge = capture_bridge
        self._pending_changes: ChangeSet | None = None
        self._closing = False

        self.title("VoiceToText Setup" if first_run else "VoiceToText Settings")
        self.geometry("760x620")
        self.protocol("WM_DELETE_WINDOW", self.close)

        self._engine_bridge = TkEngineBridge(schedule, self)
        self._state_bridge = TkStateBridge(schedule, self)
        self._download_bridge = TkDownloadBridge(schedule, self)

        self._build_ui()

        self._service.subscribe_engine(self._engine_bridge)
        self._service.subscribe_state_changes(self._state_bridge)
        self._service.subscribe_downloads(self._download_bridge)
        self._service.subscribe_model_deleted(
            self._download_bridge.on_model_deleted,
        )
        self._refresh_save_enabled()
        self.withdraw()

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
            self._sidebar.insert("end", label)
        self._sidebar.bind("<<ListboxSelect>>", self._on_sidebar_select)
        self._sidebar.pack(side="left", fill="y")

        content = ttk.Frame(body)
        content.pack(side="left", fill="both", expand=True)
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)

        draft = self._service.current_draft()
        self._transcription_page = TkTranscriptionPage(content, self._service)
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
        self._output_page = TkOutputPage(content)

        self._pages = [
            self._transcription_page,
            self._hotkeys_page,
            self._output_page,
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
            footer, text="", foreground=_STATUS_NEUTRAL_COLOR,
        )
        self._status_label.pack(side="left", fill="x", expand=True)

        button_text = "Save & Start" if self._first_run else "Save & Apply"
        self._save_button = ttk.Button(
            footer, text=button_text, command=self._on_save_clicked,
        )
        self._save_button.pack(side="right")

    def _apply_initial_draft(self, draft: SettingsDraft) -> None:
        self._hotkeys_page.set_combination(draft.hotkey_combination)
        self._hotkeys_page.set_mode(draft.hotkey_mode)
        self._output_page.set_mode(draft.output_mode)

    def _on_sidebar_select(self, _event: tkinter.Event) -> None:
        selection = self._sidebar.curselection()
        if selection:
            self._show_page(selection[0])

    def _show_page(self, index: int) -> None:
        self._pages[index].tkraise()

    def show(self) -> None:
        """Reveal and focus the window (also the single-instance focus path)."""
        self.deiconify()
        self.lift()
        self.focus_force()

    # ------------------------------------------------------------------
    # Save flow
    # ------------------------------------------------------------------

    def collect_draft(self) -> SettingsDraft:
        return SettingsDraft(
            model_size=self._transcription_page.current_model(),
            language=self._transcription_page.current_language(),
            device=self._transcription_page.current_device(),
            engine=self._transcription_page.current_engine(),
            hotkey_combination=self._hotkeys_page.current_combination(),
            hotkey_mode=self._hotkeys_page.current_mode(),
            output_mode=self._output_page.current_mode(),
            transcription_mode=self._transcription_page.current_mode(),
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
        changes = self._service.apply_draft(draft)
        self._pending_changes = changes

        if changes.engine_reload_required and not self._first_run:
            self._disable_save_button()
            self._show_status("Loading model, please wait...")
            self._service.trigger_engine_swap()
            return

        self._finalize_save()

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
            foreground=_STATUS_ERROR_COLOR if error else _STATUS_NEUTRAL_COLOR,
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

    def save_button(self) -> ttk.Button:
        return self._save_button

    def status_label(self) -> ttk.Label:
        return self._status_label

    def save_enabled(self) -> bool:
        return not self._save_button.instate(["disabled"])

    def sidebar_labels(self) -> list[str]:
        return [self._sidebar.get(i) for i in range(self._sidebar.size())]

    def _model_list(self):
        return self._transcription_page.model_list()
