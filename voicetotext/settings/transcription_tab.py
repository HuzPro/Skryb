from tkinter import Event
from typing import Callable

import customtkinter as ctk

from voicetotext.core.events import DownloadCompletedEvent, DownloadFailedEvent, DownloadProgressEvent, DownloadStateChangedEvent
from voicetotext.settings.auto_hide_scrollable_frame import (
    AutoHideScrollableFrame,
)
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.service import DownloadEventListener, SettingsService
from voicetotext.settings.settings_draft import ChangeSet
from voicetotext.settings.tk_cleanup import release_tk_variables


_SUPPORTED_LANGUAGES = [
    "auto", "en", "es", "fr", "de", "it", "pt", "nl", "ja", "ko", "zh",
]
_SUPPORTED_DEVICES = ["auto", "cuda", "cpu"]


class TranscriptionTab(DownloadEventListener):
    """View for picking the active Whisper model, language, and device."""

    def __init__(self, parent: ctk.CTkFrame, service: SettingsService) -> None:
        self._parent = parent
        self._service = service
        self._model_rows: dict[str, _ModelRow] = {}
        draft = service.current_draft()
        self._active_model_var = ctk.StringVar(value=draft.model_size)
        self._build(parent, draft)
        self._service.subscribe_downloads(self)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self, parent: ctk.CTkFrame, draft: ChangeSet) -> None:
        models_label = ctk.CTkLabel(
            parent, text="Models", font=ctk.CTkFont(size=15, weight="bold")
        )
        models_label.pack(anchor="w", padx=16, pady=(16, 6))

        # Scrollable frame so the list degrades gracefully when the window
        # is shrunk below the height needed to show every row, without
        # pushing the Settings section + Save button off-screen. The
        # adapter variant hides its scrollbar when the content already
        # fits — otherwise the bar stays visible at larger window heights.
        self._models_frame = AutoHideScrollableFrame(
            parent, fg_color="transparent"
        )
        self._models_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self._models_frame.bind("<Configure>", self._on_models_frame_configure)

        self._populate_model_rows()

        settings_label = ctk.CTkLabel(
            parent, text="Settings", font=ctk.CTkFont(size=15, weight="bold")
        )
        settings_label.pack(anchor="w", padx=16, pady=(6, 6))

        settings_frame = ctk.CTkFrame(parent, fg_color="transparent")
        settings_frame.pack(fill="x", padx=16)

        lang_label = ctk.CTkLabel(settings_frame, text="Language:")
        lang_label.grid(row=0, column=0, sticky="w", pady=4)

        self._language_var = ctk.StringVar(value=draft.language)
        language_menu = ctk.CTkOptionMenu(
            settings_frame,
            variable=self._language_var,
            values=_SUPPORTED_LANGUAGES,
            width=160,
        )
        language_menu.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=4)

        device_label = ctk.CTkLabel(settings_frame, text="Device:")
        device_label.grid(row=1, column=0, sticky="w", pady=4)

        self._device_var = ctk.StringVar(value=draft.device)
        device_menu = ctk.CTkOptionMenu(
            settings_frame,
            variable=self._device_var,
            values=_SUPPORTED_DEVICES,
            width=160,
        )
        device_menu.grid(row=1, column=1, sticky="w", padx=(12, 0), pady=4)

    def _populate_model_rows(self) -> None:
        for widget in self._models_frame.winfo_children():
            widget.destroy()
        self._model_rows.clear()

        for view in self._service.list_models():
            row = _ModelRow(
                parent=self._models_frame,
                view=view,
                active_var=self._active_model_var,
                on_download=self._on_download,
                on_pause=self._on_pause,
                on_resume=self._on_resume,
                on_cancel=self._on_cancel,
                on_delete=self._on_delete,
            )
            self._model_rows[view.name] = row

    # ------------------------------------------------------------------
    # Responsive layout
    # ------------------------------------------------------------------

    # Horizontal budget consumed by the radio indent, action button
    # column, and internal padding on each row. Subtracted from the
    # frame width to produce the description label's wraplength.
    _ROW_CHROME_WIDTH = 180
    _MIN_DESCRIPTION_WRAPLENGTH = 280

    def _on_models_frame_configure(self, event: Event) -> None:
        wraplength = max(
            self._MIN_DESCRIPTION_WRAPLENGTH,
            event.width - self._ROW_CHROME_WIDTH,
        )
        for row in self._model_rows.values():
            row.set_description_wraplength(wraplength)

    # ------------------------------------------------------------------
    # Commands (called from UI thread)
    # ------------------------------------------------------------------

    def _on_download(self, model_name: str) -> None:
        row = self._model_rows.get(model_name)
        if row is not None:
            row.set_queued()
        self._service.start_download(model_name)

    def _on_pause(self, model_name: str) -> None:
        try:
            self._service.pause_download(model_name)
        except LookupError:
            pass

    def _on_resume(self, model_name: str) -> None:
        row = self._model_rows.get(model_name)
        if row is not None:
            row.set_queued()
        self._service.resume_download(model_name)

    def _on_cancel(self, model_name: str) -> None:
        try:
            self._service.cancel_download(model_name)
        except LookupError:
            pass
        row = self._model_rows.get(model_name)
        if row is not None:
            row.set_not_installed()

    def _on_delete(self, model_name: str) -> None:
        try:
            self._service.delete_model(model_name)
        except ValueError:
            # Silently ignore deletes of the active model — the button
            # should not have been offered in the first place.
            return
        row = self._model_rows.get(model_name)
        if row is not None:
            row.set_not_installed()

    # ------------------------------------------------------------------
    # Draft accessors
    # ------------------------------------------------------------------

    def current_model(self) -> str:
        return self._active_model_var.get()

    def current_language(self) -> str:
        return self._language_var.get()

    def current_device(self) -> str:
        return self._device_var.get()

    # ------------------------------------------------------------------
    # DownloadEventListener
    # ------------------------------------------------------------------

    def dispose(self) -> None:
        self._service.unsubscribe_downloads(self)
        # Break tkinter Variable → Tk references so their ``__del__``
        # doesn't try to query a dead interpreter during GC. Each settings
        # open runs on a fresh daemon thread with its own Tk instance; the
        # Variables can outlive that instance and raise
        # "main thread is not in main loop" otherwise.
        release_tk_variables(
            self._active_model_var,
            self._language_var,
            self._device_var,
        )

    def on_download_progress(self, event: DownloadProgressEvent) -> None:
        self._parent.after(0, self._apply_progress, event)

    def on_download_state(self, event: DownloadStateChangedEvent) -> None:
        self._parent.after(0, self._apply_state, event)

    def on_download_completed(self, event: DownloadCompletedEvent) -> None:
        self._parent.after(0, self._apply_completed, event)

    def on_download_failed(self, event: DownloadFailedEvent) -> None:
        self._parent.after(0, self._apply_failed, event)

    def _apply_progress(self, event: DownloadProgressEvent) -> None:
        row = self._model_rows.get(event.job_id)
        if row is not None:
            row.set_progress(
                bytes_done=event.bytes_done,
                bytes_total=event.bytes_total,
                speed_bps=event.speed_bps,
                eta_seconds=event.eta_seconds,
            )

    def _apply_state(self, event: DownloadStateChangedEvent) -> None:
        row = self._model_rows.get(event.job_id)
        if row is None:
            return
        if event.state == "downloading":
            row.set_downloading()
        elif event.state == "paused":
            row.set_paused()

    def _apply_completed(self, event: DownloadCompletedEvent) -> None:
        row = self._model_rows.get(event.job_id)
        if row is not None:
            row.set_installed()

    def _apply_failed(self, event: DownloadFailedEvent) -> None:
        row = self._model_rows.get(event.job_id)
        if row is not None:
            row.set_error(event.error)


class _ModelRow:
    """One row in the Models list — stacked two-line layout.

    Top line: radio button with model name and size, action button on the right.
    Bottom line: wrapped description (never clipped).
    """

    _ModelCommand = Callable[[str], None]

    def __init__(
        self,
        parent: ctk.CTkFrame,
        view: ModelView,
        active_var: ctk.StringVar,
        on_download: _ModelCommand,
        on_pause: _ModelCommand,
        on_resume: _ModelCommand,
        on_cancel: _ModelCommand,
        on_delete: _ModelCommand,
    ) -> None:
        self._view = view
        self._active_var = active_var
        self._on_download = on_download
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._on_cancel = on_cancel
        self._on_delete = on_delete

        self._frame = ctk.CTkFrame(parent, corner_radius=6)
        self._frame.pack(fill="x", pady=3)
        self._frame.grid_columnconfigure(0, weight=1)

        self._radio = ctk.CTkRadioButton(
            self._frame,
            text=f"{view.name}   {view.display_size}",
            variable=active_var,
            value=view.name,
            font=ctk.CTkFont(size=13),
        )
        self._radio.grid(row=0, column=0, sticky="w", padx=(12, 10), pady=(10, 0))
        if not view.installed:
            self._radio.configure(state="disabled")

        self._desc_label = ctk.CTkLabel(
            self._frame,
            text=view.description,
            font=ctk.CTkFont(size=11),
            text_color="gray",
            wraplength=360,
            justify="left",
            anchor="w",
        )
        self._desc_label.grid(
            row=1, column=0, sticky="w", padx=(36, 10), pady=(2, 10)
        )

        self._action_frame = ctk.CTkFrame(self._frame, fg_color="transparent")
        self._action_frame.grid(
            row=0, column=1, rowspan=2, sticky="e", padx=12, pady=10
        )

        self._progress_frame = ctk.CTkFrame(self._frame, fg_color="transparent")
        self._progress_frame.grid_columnconfigure(0, weight=1)
        self._progress_bar: ctk.CTkProgressBar | None = None
        self._status_label: ctk.CTkLabel | None = None
        self._progress_visible = False

        if view.installed:
            self._show_installed_buttons()
        else:
            self._show_download_button()

    # ------------------------------------------------------------------
    # Action buttons
    # ------------------------------------------------------------------

    def _clear_action_frame(self) -> None:
        for widget in self._action_frame.winfo_children():
            widget.destroy()

    def _clear_progress_frame(self) -> None:
        for widget in self._progress_frame.winfo_children():
            widget.destroy()
        self._progress_bar = None
        self._status_label = None
        if self._progress_visible:
            self._progress_frame.grid_remove()
            self._progress_visible = False

    def _show_progress_frame(self) -> None:
        if not self._progress_visible:
            self._progress_frame.grid(
                row=2, column=0, columnspan=2, padx=12, pady=(0, 10), sticky="we"
            )
            self._progress_visible = True

    def _show_download_button(self) -> None:
        self._clear_action_frame()
        btn = ctk.CTkButton(
            self._action_frame,
            text="Download",
            width=88,
            height=30,
            command=self._wrap_command(self._on_download),
        )
        btn.pack()

    def _show_installed_buttons(self) -> None:
        self._clear_action_frame()
        delete_btn = ctk.CTkButton(
            self._action_frame,
            text="Delete",
            width=76,
            height=30,
            fg_color="transparent",
            border_width=1,
            text_color=("gray30", "gray70"),
            command=self._wrap_command(self._on_delete),
        )
        delete_btn.pack()

    def _show_pause_cancel_buttons(self) -> None:
        self._clear_action_frame()
        pause_btn = ctk.CTkButton(
            self._action_frame,
            text="Pause",
            width=64,
            height=30,
            command=self._wrap_command(self._on_pause),
        )
        pause_btn.pack(side="left", padx=(0, 4))
        cancel_btn = ctk.CTkButton(
            self._action_frame,
            text="Cancel",
            width=64,
            height=30,
            fg_color="transparent",
            border_width=1,
            text_color=("gray30", "gray70"),
            command=self._wrap_command(self._on_cancel),
        )
        cancel_btn.pack(side="left")

    def _show_resume_cancel_buttons(self) -> None:
        self._clear_action_frame()
        resume_btn = ctk.CTkButton(
            self._action_frame,
            text="Resume",
            width=72,
            height=30,
            command=self._wrap_command(self._on_resume),
        )
        resume_btn.pack(side="left", padx=(0, 4))
        cancel_btn = ctk.CTkButton(
            self._action_frame,
            text="Cancel",
            width=64,
            height=30,
            fg_color="transparent",
            border_width=1,
            text_color=("gray30", "gray70"),
            command=self._wrap_command(self._on_cancel),
        )
        cancel_btn.pack(side="left")

    def _wrap_command(self, handler: Callable[[str], None]) -> Callable[[], None]:
        model_name = self._view.name
        return lambda: handler(model_name)

    def set_description_wraplength(self, pixels: int) -> None:
        self._desc_label.configure(wraplength=pixels)

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _ensure_progress_widgets(self) -> None:
        self._show_progress_frame()
        if self._progress_bar is None:
            self._progress_bar = ctk.CTkProgressBar(self._progress_frame)
            self._progress_bar.set(0)
            self._progress_bar.grid(row=0, column=0, sticky="we", padx=(0, 10))
        if self._status_label is None:
            self._status_label = ctk.CTkLabel(
                self._progress_frame, text="", font=ctk.CTkFont(size=11)
            )
            self._status_label.grid(row=0, column=1, sticky="e")

    def set_queued(self) -> None:
        self._ensure_progress_widgets()
        self._progress_bar.set(0)
        self._status_label.configure(text="queued…")
        self._show_pause_cancel_buttons()

    def set_downloading(self) -> None:
        self._ensure_progress_widgets()
        self._show_pause_cancel_buttons()

    def set_paused(self) -> None:
        self._show_resume_cancel_buttons()
        if self._status_label is not None:
            self._status_label.configure(text="paused")

    def set_progress(
        self,
        bytes_done: int,
        bytes_total: int,
        speed_bps: float,
        eta_seconds: float | None,
    ) -> None:
        self._ensure_progress_widgets()
        fraction = bytes_done / bytes_total if bytes_total > 0 else 0.0
        self._progress_bar.set(min(fraction, 1.0))
        pct = int(fraction * 100)
        speed_text = _format_speed(speed_bps)
        eta_text = _format_eta(eta_seconds)
        self._status_label.configure(text=f"{pct}% · {speed_text} · {eta_text}")

    def set_installed(self) -> None:
        self._clear_progress_frame()
        self._radio.configure(state="normal")
        self._show_installed_buttons()

    def set_not_installed(self) -> None:
        self._clear_progress_frame()
        self._radio.configure(state="disabled")
        self._show_download_button()

    def set_error(self, message: str) -> None:
        self._ensure_progress_widgets()
        display = message if len(message) <= 60 else message[:57] + "..."
        self._status_label.configure(text=f"error: {display}", text_color="#FF6666")
        self._clear_action_frame()
        retry_btn = ctk.CTkButton(
            self._action_frame,
            text="Retry",
            width=64,
            height=30,
            command=self._wrap_command(self._on_download),
        )
        retry_btn.pack(side="left", padx=(0, 4))
        dismiss_btn = ctk.CTkButton(
            self._action_frame,
            text="Dismiss",
            width=72,
            height=30,
            fg_color="transparent",
            border_width=1,
            text_color=("gray30", "gray70"),
            command=self._wrap_command(self._on_cancel),
        )
        dismiss_btn.pack(side="left")


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------


def _format_speed(bps: float) -> str:
    if bps <= 0:
        return "— B/s"
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if bps < 1024:
            return f"{bps:.1f} {unit}"
        bps /= 1024
    return f"{bps:.1f} TB/s"


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    if seconds >= 3600:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h{mins:02d}m"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"
