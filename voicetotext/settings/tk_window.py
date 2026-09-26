import re
import sys
import tkinter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, ttk

from voicetotext.core.color_scheme import ColorScheme, detect_app_color_scheme
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
from voicetotext.core.work_area import foreground_work_area
from voicetotext.settings.download_progress import DownloadProgressSnapshot
from voicetotext.settings.license_consent import (
    LicenseConsentPrompt,
    ask_license_consent,
    collect_license_acceptance,
)
from voicetotext.settings.service import SettingsService
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.tk_discard_prompt import (
    UnsavedChangesPrompt,
    ask_save_discard_cancel,
)
from voicetotext.settings.tk_download_bridge import TkDownloadBridge
from voicetotext.settings.tk_engine_bridge import TkEngineBridge
from voicetotext.settings.tk_hotkey_capture import CaptureBridge
from voicetotext.recording.input_devices import Microphones
from voicetotext.settings.tk_pages.audio_page import TkAudioPage
from voicetotext.settings.tk_pages.dictionary_page import TkDictionaryPage
from voicetotext.settings.tk_pages.history_page import TkHistoryPage
from voicetotext.settings.tk_pages.hotkeys_page import TkHotkeysPage
from voicetotext.settings.tk_pages.output_page import TkOutputPage
from voicetotext.settings.tk_pages.transcription_page import TkTranscriptionPage
from voicetotext.settings.tk_pages.writing_page import TkWritingPage
from voicetotext.settings.tk_sidebar import TkSidebar
from voicetotext.settings.tk_state_bridge import TkStateBridge
from voicetotext.settings.theme import (
    CONTROL_GAP,
    DIVIDER_STYLE,
    MUTED_STYLE,
    PAGE_PADDING,
    PRIMARY_BUTTON_STYLE,
    ScrollArea,
    Theme,
    apply_theme,
    divider,
    theme_for,
)
from voicetotext.settings.win32_ime_composition_hook import (
    Win32ImeCompositionHook,
)
from voicetotext.settings.win32_titlebar import show_with_titlebar
from voicetotext.writing.service import WritingService

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]
ChangeSetCallback = Callable[[ChangeSet], None]
CaptureStateCallback = Callable[[bool], None]

# Key, sidebar glyph, title. The glyphs are ones Segoe UI Symbol draws in
# a single colour.
_SECTIONS = (
    ("transcription", "◉", "Transcription"),
    ("audio", "♫", "Microphone & sounds"),
    ("shortcuts", "⌨", "Shortcuts"),
    ("output", "⇥", "Output"),
    ("writing", "✎", "Writing"),
    ("history", "↺", "History"),
    ("dictionary", "≡", "Dictionary"),
)
_SIZE = (980, 800)
# Tk's geometry leaves the title bar out of the height.
_CAPTION_ALLOWANCE = 40
_WHEEL_NOTCH = 120
_POSITION = re.compile(r"([+-]-?\d+)([+-]-?\d+)$")
# Where the window was last closed, so reopening it in the same session
# puts it back instead of recentring it.
_session_position: tuple[int, int] | None = None


def window_placement(
    work_area: tuple[int, int, int, int], remembered: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    """Width, height, x, y: centred in the work area, or where it was closed.

    Either way the whole window, footer included, stays inside the work
    area; Windows' own cascade can leave the footer under the taskbar.
    """
    left, top, right, bottom = work_area
    width = min(_SIZE[0], right - left)
    height = min(_SIZE[1], bottom - top - _CAPTION_ALLOWANCE)
    x, y = remembered or (
        (left + right - width) // 2,
        (top + bottom - height - _CAPTION_ALLOWANCE) // 2,
    )
    x = max(left, min(x, right - width))
    y = max(top, min(y, bottom - height - _CAPTION_ALLOWANCE))
    return width, height, x, y


def _run_now(action: Callable[[], None]) -> None:
    action()


def _ignore_changes(_changes: ChangeSet) -> None:
    pass


def _ignore_capture(_recording: bool) -> None:
    pass


class TkSettingsWindow(tkinter.Toplevel):
    """Top-level Tk settings window: sidebar nav + scrolling pages + footer.

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
        writing: WritingService | None = None,
        delete_all_history: Callable[[], None] | None = None,
        ask_unsaved: UnsavedChangesPrompt = ask_save_discard_cancel,
    ) -> None:
        super().__init__(master)
        # The first-run window has no writing model to manage yet.
        self._writing = writing
        self._delete_all_history = delete_all_history
        self._ask_unsaved = ask_unsaved
        self._microphones = microphones or Microphones()
        self._shown_page: ttk.Frame | None = None
        self._shown_area: ScrollArea | None = None
        self._placed = False
        # Hide immediately so the half-built, not-yet-themed window never
        # flashes on screen; ``show`` reveals it once it is themed.
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
        # ``None`` means follow the Windows app light/dark setting at open time.
        if color_scheme is None:
            color_scheme = detect_app_color_scheme()
        self._theme: Theme = theme_for(color_scheme)

        self.title("VoiceToText Setup" if first_run else "VoiceToText Settings")
        self.geometry("{}x{}".format(*_SIZE))
        self.minsize(860, 640)
        self.protocol("WM_DELETE_WINDOW", self._on_close_requested)

        self._engine_bridge = TkEngineBridge(schedule, self)
        self._state_bridge = TkStateBridge(schedule, self)
        self._download_bridge = TkDownloadBridge(schedule, self)

        self._build_ui()
        apply_theme(self, self._theme)
        self._scroll_pages_with_the_wheel()
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
        divider(self).pack(side="bottom", fill="x")

        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)
        content = ttk.Frame(body)
        content.rowconfigure(0, weight=1)
        content.columnconfigure(0, weight=1)
        sections = [
            section for section in _SECTIONS
            if section[0] != "writing" or self._writing is not None
        ]
        areas = {key: ScrollArea(content, title) for key, _glyph, title in sections}

        draft = self._service.current_draft()
        self._transcription_page = TkTranscriptionPage(
            areas["transcription"].body, self._service,
            confirm_license=self._confirm_license, schedule=self._schedule,
        )
        self._transcription_page.download_error.connect(self._on_download_error)
        self._transcription_page.delete_error.connect(self._on_delete_error)
        self._transcription_page.model_auto_switched.connect(
            self._on_model_auto_switched,
        )
        self._hotkeys_page = TkHotkeysPage(
            areas["shortcuts"].body, self._capture_bridge, self._schedule,
        )
        self._hotkeys_page.recording_state_changed.connect(
            self._on_recording_state_changed,
        )
        self._audio_page = TkAudioPage(areas["audio"].body, self._microphones)
        self._output_page = TkOutputPage(
            areas["output"].body,
            on_reset_pill_position=self._service.reset_pill_position,
        )
        self._history_page = TkHistoryPage(
            areas["history"].body,
            logs_folder=str(self._service.diagnostics_directory()),
            delete_all_history=(
                None if self._delete_all_history is None
                else self._on_delete_all_history
            ),
        )
        self._dictionary_page = TkDictionaryPage(areas["dictionary"].body)
        self._dictionary_page.set_dictionary(self._service.dictionary_rules())
        self._writing_page = (
            TkWritingPage(areas["writing"].body, self._writing, self._schedule)
            if self._writing is not None
            else None
        )

        pages: dict[str, ttk.Frame | None] = {
            "transcription": self._transcription_page,
            "audio": self._audio_page,
            "shortcuts": self._hotkeys_page,
            "output": self._output_page,
            "writing": self._writing_page,
            "history": self._history_page,
            "dictionary": self._dictionary_page,
        }
        self._pages: list[ttk.Frame] = []
        self._areas: list[ScrollArea] = []
        for key, _glyph, _title in sections:
            page = pages[key]
            if page is None:
                continue
            page.pack(fill="x")
            areas[key].grid(row=0, column=0, sticky="nsew")
            self._pages.append(page)
            self._areas.append(areas[key])

        self._sidebar = TkSidebar(
            body, [(glyph, title) for _key, glyph, title in sections], self._show_page,
        )
        self._sidebar.pack(side="left", fill="y")
        ttk.Frame(body, style=DIVIDER_STYLE, width=1).pack(side="left", fill="y")
        content.pack(side="left", fill="both", expand=True)

        self._apply_initial_draft(draft)
        self._saved_draft = self.collect_draft()
        self._show_page(0)

    def _build_footer(self) -> None:
        footer = ttk.Frame(self, padding=(PAGE_PADDING, 12))
        footer.pack(side="bottom", fill="x")

        button_text = "Save & Start" if self._first_run else "Save & Apply"
        self._save_button = ttk.Button(
            footer, text=button_text, style=PRIMARY_BUTTON_STYLE,
            command=self._on_save_clicked,
        )
        self._save_button.pack(side="right")
        self._export_settings_button = ttk.Button(
            footer, text="Export...", command=self._on_export_settings,
        )
        self._import_settings_button = ttk.Button(
            footer, text="Import...", command=self._on_import_settings,
        )
        self._logs_folder_button = ttk.Button(
            footer, text="Logs folder...", command=self._on_choose_logs_folder,
        )
        for button in (
            self._export_settings_button,
            self._import_settings_button,
            self._logs_folder_button,
        ):
            button.pack(side="left", padx=(0, CONTROL_GAP))
        self._status_label = ttk.Label(
            footer, text="", style=MUTED_STYLE, justify="left",
            foreground=self._theme.status_neutral,
        )
        self._status_label.pack(side="left", fill="x", expand=True, padx=(8, 16))
        self._status_label.bind(
            "<Configure>",
            lambda event: self._status_label.configure(wraplength=max(event.width, 120)),
        )

    def _apply_initial_draft(self, draft: SettingsDraft) -> None:
        self._hotkeys_page.set_bindings(draft.hotkey_bindings)
        self._hotkeys_page.set_double_tap_locks_hold(draft.double_tap_locks_hold)
        self._output_page.set_mode(draft.output_mode)
        self._history_page.set_capture_content(draft.capture_content)
        self._history_page.set_history_retention_days(draft.history_retention_days)
        self._output_page.set_paste_method_overrides(draft.paste_method_overrides)
        self._audio_page.set_device(draft.input_device)
        self._audio_page.set_boost_quiet_speech(draft.boost_quiet_speech)
        self._audio_page.set_recording_preferences(draft.recording)
        if self._writing_page is not None:
            self._writing_page.set_settings(draft.writing)

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

    def _on_delete_all_history(self) -> None:
        if self._delete_all_history is None:
            return
        self._delete_all_history()
        self._show_status("Deleting all history in the background.")

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
        self._saved_draft = self.collect_draft()
        self._show_status("Settings imported.")

    def _apply_imported_draft(self) -> None:
        draft = self._service.current_draft()
        self._transcription_page.apply_mode(draft.transcription_mode)
        self._transcription_page.set_model(draft.model_size)
        self._transcription_page.set_language(draft.language)
        self._transcription_page.set_engine(draft.engine)
        self._transcription_page.set_crisperwhisper_mode(draft.crisperwhisper_mode)
        self._apply_initial_draft(draft)

    def _show_page(self, index: int) -> None:
        page = self._pages[index]
        if self._shown_page is self._audio_page and page is not self._audio_page:
            self._audio_page.on_hidden()
        self._areas[index].tkraise()
        self._sidebar.select(index)
        if page is self._audio_page and self._shown_page is not page:
            self._audio_page.on_shown()
        self._shown_page = page
        self._shown_area = self._areas[index]

    def _scroll_pages_with_the_wheel(self) -> None:
        # Over a combobox or spinbox the wheel would change its value rather
        # than scroll the page, changing settings by accident on the way
        # down a long page; both are set by clicking instead.
        self.bind_class("TCombobox", "<MouseWheel>", "")
        self.bind_class("TSpinbox", "<MouseWheel>", "")
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Control-Tab>", lambda _event: self._step_page(1))
        self.bind("<Control-Shift-Tab>", lambda _event: self._step_page(-1))

    def _on_mousewheel(self, event: tkinter.Event) -> None:
        if self._shown_area is None or isinstance(
            event.widget, (tkinter.Text, tkinter.Listbox, ttk.Treeview),
        ):
            return
        notches = -event.delta / _WHEEL_NOTCH
        self._shown_area.scroll(round(notches) or (1 if notches > 0 else -1))

    def _step_page(self, step: int) -> None:
        current = self._pages.index(self._shown_page) if self._shown_page else 0
        self._show_page((current + step) % len(self._pages))

    def show(self) -> None:
        """Reveal and focus the window (also the single-instance focus path)."""
        if not self._placed:
            self._place_on_screen()
            self._placed = True
        show_with_titlebar(self, dark=self._theme.is_dark)
        self.lift()
        self.focus_force()

    def _place_on_screen(self) -> None:
        width, height, x, y = window_placement(self._work_area(), _session_position)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _work_area(self) -> tuple[int, int, int, int]:
        try:
            return foreground_work_area()
        except OSError:
            return 0, 0, self.winfo_screenwidth(), self.winfo_screenheight()

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
            output_mode=self._output_page.current_mode(),
            capture_content=self._history_page.captures_content(),
            input_device=self._audio_page.current_device(),
            boost_quiet_speech=self._audio_page.boosts_quiet_speech(),
            recording=self._audio_page.recording_preferences(),
            history_retention_days=self._history_page.history_retention_days(),
            paste_method_overrides=self._output_page.paste_method_overrides(),
            transcription_mode=self._transcription_page.current_mode(),
            crisperwhisper_mode=(
                self._transcription_page.current_crisperwhisper_mode()
            ),
            writing=(
                self._writing_page.settings()
                if self._writing_page is not None
                else self._service.current_draft().writing
            ),
            hotkey_bindings=self._hotkeys_page.bindings(),
            double_tap_locks_hold=self._hotkeys_page.double_tap_locks_hold(),
        )

    def has_unsaved_changes(self) -> bool:
        return (
            self.collect_draft() != self._saved_draft
            or self._dictionary_page.is_dirty()
            or (self._writing_page is not None and self._writing_page.transforms_dirty())
        )

    def _on_close_requested(self) -> None:
        if not self.has_unsaved_changes():
            self.close()
            return
        answer = self._ask_unsaved(self, self._theme)
        if answer == "save":
            self._on_save_clicked()
        elif answer == "discard":
            self.close()

    def _on_save_clicked(self) -> None:
        if self._first_run and not self._service.any_model_installed():
            self._show_status(
                "Please download at least one model before saving.",
                error=True,
            )
            return

        shortcut_problem = self._hotkeys_page.validation_error()
        if shortcut_problem:
            self._show_status(shortcut_problem, error=True)
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
        self._pending_changes = self._save_transform_edits(
            self._save_dictionary_edits(changes),
        )

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
        if self._writing is not None:
            self._writing.forget_learned_words(
                self._dictionary_page.forgotten_learned_words(),
            )
        return replace(changes, output_changed=True)

    def _save_transform_edits(self, changes: ChangeSet) -> ChangeSet:
        if self._writing is None or self._writing_page is None:
            return changes
        if not self._writing_page.transforms_dirty():
            return changes
        self._writing.save_transforms(self._writing_page.transforms())
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
        self._remember_position()
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

    def _remember_position(self) -> None:
        global _session_position
        position = _POSITION.search(self.wm_geometry())
        if self._placed and position:
            _session_position = (int(position.group(1)), int(position.group(2)))

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

    def history_page(self) -> TkHistoryPage:
        return self._history_page

    def dictionary_page(self) -> TkDictionaryPage:
        return self._dictionary_page

    def writing_page(self) -> TkWritingPage | None:
        return self._writing_page

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
        return self._sidebar.labels()

    def _model_list(self):
        return self._transcription_page.model_list()
