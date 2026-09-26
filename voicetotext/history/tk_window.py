import logging
import threading
import tkinter
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from voicetotext.core.color_scheme import detect_app_color_scheme
from voicetotext.core.timestamps import utc_now
from voicetotext.history.entries import (
    CANCELLED,
    EMPTY,
    FAILED,
    NOT_SAVED,
    HistoryEntry,
    empty_state_message,
    filter_entries,
    load_history,
    local_time,
    preview,
)
from voicetotext.history.management import (
    default_export_name,
    delete_all_history,
    delete_entries,
    export_audio,
)
from voicetotext.recording.audio_segment import load_wav
from voicetotext.settings.theme import (
    BODY_FONT,
    CARD_MUTED_STYLE,
    CONTROL_GAP,
    DANGER_BUTTON_STYLE,
    MUTED_STYLE,
    PAGE_TEXT_WRAP,
    Theme,
    apply_theme,
    card,
    theme_for,
)
from voicetotext.settings.win32_titlebar import show_with_titlebar
from voicetotext.tray.duration_format import format_duration

_logger = logging.getLogger(__name__)

ScheduleOnGuiThread = Callable[[Callable[[], None]], None]
RetryOutcome = str | Exception
Retranscribe = Callable[[Path, Callable[[RetryOutcome], None]], None]

_COLUMNS = (
    ("time", "When", 130, False),
    ("text", "Text", 420, True),
    ("duration", "Length", 64, False),
    ("status", "Status", 84, False),
)
_ROW_TAGS = {FAILED: "failed", CANCELLED: "quiet", EMPTY: "quiet", NOT_SAVED: "quiet"}
_TRANSCRIBING = "Transcribing…"
_NO_MATCHES = "No dictations match your search."
_PADDING = 16


def _start_daemon_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="history-load", daemon=True).start()


def _play_wav(path: Path) -> float:
    """Start playing *path* on the default output device; returns its length."""
    audio = load_wav(path)
    import sounddevice as sd

    sd.play(audio.data, audio.sample_rate)
    return audio.duration_seconds


def _stop_playback() -> None:
    import sounddevice as sd

    sd.stop()


class HistoryWindow:
    """Past dictations read back from the diagnostics logs.

    One instance lives for the app's lifetime; :meth:`show` builds the
    window on first use and afterwards just raises it. The logs are read
    on a background thread (they grow without bound) and the result hops
    back to the GUI thread through ``schedule``, as does every
    ``retranscribe`` outcome. Retried text is kept in memory only.
    """

    def __init__(
        self,
        root: tkinter.Misc,
        diagnostics_dir: Path,
        retranscribe: Retranscribe,
        schedule: ScheduleOnGuiThread,
        *,
        run_in_background: Callable[[Callable[[], None]], None] = (
            _start_daemon_thread
        ),
    ) -> None:
        self._root = root
        self._diagnostics_dir = diagnostics_dir
        self._retranscribe = retranscribe
        self._schedule = schedule
        self._run_in_background = run_in_background
        self._window: tkinter.Toplevel | None = None
        self._entries: list[HistoryEntry] = []
        self._shown: list[HistoryEntry] = []
        self._hint_text = ""
        # ``None`` marks a retry still running.
        self._retries: dict[tuple[str, int], RetryOutcome | None] = {}
        self._playback_end: str | None = None

    def show(self) -> None:
        """Open the window, or raise it if it is already open; then re-read the logs."""
        if self._window is None:
            self._build()
        window = self._window
        show_with_titlebar(window, dark=self._theme.is_dark)
        window.lift()
        window.focus_force()
        self.refresh()

    def refresh(self) -> None:
        if self._window is None:
            return
        self._hint_text = "Loading…"
        self._show_hint(self._hint_text)
        self._run_in_background(self._load)

    def close(self) -> None:
        if self._window is None:
            return
        self._stop()
        self._window.destroy()
        self._window = None

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        window = tkinter.Toplevel(self._root)
        # Hidden until ``show`` so the unthemed window never flashes.
        window.withdraw()
        window.title("VoiceToText History")
        window.geometry("820x640")
        window.minsize(560, 420)
        window.protocol("WM_DELETE_WINDOW", self.close)
        self._window = window
        self._theme: Theme = theme_for(detect_app_color_scheme())

        self._build_toolbar(window)
        self._build_detail(window)
        self._build_list(window)
        apply_theme(window, self._theme)
        self._tree.tag_configure("failed", foreground=self._theme.danger)
        self._tree.tag_configure("quiet", foreground=self._theme.muted)
        self._render_selection()

    def _build_toolbar(self, window: tkinter.Toplevel) -> None:
        toolbar = ttk.Frame(window)
        toolbar.pack(side="top", fill="x", padx=_PADDING, pady=(_PADDING, 10))
        ttk.Label(toolbar, text="Search", style=MUTED_STYLE).pack(side="left")
        self._query = tkinter.StringVar(master=window)
        self._query.trace_add("write", lambda *_args: self._show_filtered())
        self._search = ttk.Entry(toolbar, textvariable=self._query, font=BODY_FONT)
        self._search.pack(side="left", fill="x", expand=True, padx=(CONTROL_GAP, 16))
        self._delete_all_button = ttk.Button(
            toolbar, text="Delete all", style=DANGER_BUTTON_STYLE, command=self._delete_all,
        )
        self._delete_all_button.pack(side="right")
        self._delete_button = ttk.Button(toolbar, text="Delete", command=self._delete)
        self._delete_button.pack(side="right", padx=CONTROL_GAP)
        self._export_button = ttk.Button(
            toolbar, text="Save audio as...", command=self._export,
        )
        self._export_button.pack(side="right")
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(
            side="right", padx=(0, CONTROL_GAP),
        )

    def _build_list(self, window: tkinter.Toplevel) -> None:
        frame = card(window, padding=1)
        frame.pack(side="top", fill="both", expand=True, padx=_PADDING)
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        self._tree = ttk.Treeview(
            frame,
            columns=[name for name, *_rest in _COLUMNS],
            show="headings",
            selectmode="browse",
        )
        for name, heading, width, stretch in _COLUMNS:
            self._tree.heading(name, text=heading, anchor="w")
            self._tree.column(name, width=width, stretch=stretch, anchor="w")
        self._scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=self._scroll_list)
        self._tree.grid(row=1, column=0, sticky="nsew")
        self._tree.bind("<<TreeviewSelect>>", lambda _event: self._render_selection())
        self._hint = ttk.Label(
            frame, text="", style=CARD_MUTED_STYLE, wraplength=PAGE_TEXT_WRAP,
            justify="center",
        )

    def _scroll_list(self, first: str, last: str) -> None:
        # With nothing to scroll the bar would be a thumb filling its trough.
        if float(first) <= 0.0 and float(last) >= 1.0:
            self._scrollbar.grid_remove()
        else:
            self._scrollbar.grid(row=1, column=1, sticky="ns")
        self._scrollbar.set(first, last)

    def _build_detail(self, window: tkinter.Toplevel) -> None:
        bottom = ttk.Frame(window)
        bottom.pack(side="bottom", fill="x", padx=_PADDING, pady=(12, _PADDING))
        self._detail = tkinter.Text(
            bottom, height=6, wrap="word", state="disabled",
        )
        self._detail.pack(side="top", fill="x")
        actions = ttk.Frame(bottom)
        actions.pack(side="top", fill="x", pady=(10, 0))
        self._copy_button = ttk.Button(actions, text="Copy", command=self._copy)
        self._copy_button.pack(side="left")
        self._play_button = ttk.Button(actions, text="Play", command=self._toggle_play)
        self._play_button.pack(side="left", padx=CONTROL_GAP)
        self._retry_button = ttk.Button(actions, text="Retry", command=self._retry)
        self._retry_button.pack(side="left")
        self._status = ttk.Label(bottom, text="", style=MUTED_STYLE, justify="left")
        self._status.pack(side="top", fill="x", pady=(6, 0))
        self._status.bind(
            "<Configure>",
            lambda event: self._status.configure(wraplength=max(event.width, 200)),
        )

    def _show_hint(self, text: str) -> None:
        """Say why the list is empty in the middle of it, or anything else above it."""
        if not text and self._entries and not self._shown:
            text = _NO_MATCHES
        self._hint.configure(text=text)
        if not text:
            self._hint.grid_remove()
            self._hint.place_forget()
        elif self._tree.get_children():
            self._hint.place_forget()
            self._hint.grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=8)
        else:
            self._hint.grid_remove()
            self._hint.place(in_=self._tree, relx=0.5, rely=0.5, anchor="center")

    # ------------------------------------------------------------------
    # Loading and listing
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            entries = load_history(self._diagnostics_dir)
        except Exception:  # a bad log must not leave "Loading..." up
            _logger.warning("History load failed", exc_info=True)
            entries = []
        self._schedule(lambda: self._show_entries(entries))

    def _show_entries(self, entries: list[HistoryEntry]) -> None:
        if self._window is None:
            return
        self._entries = entries
        self._hint_text = empty_state_message(entries, self._diagnostics_dir)
        self._show_filtered()

    def _show_filtered(self) -> None:
        # ponytail: rebuilds every row per keystroke (~150 ms at 4k rows);
        # debounce the search box if history grows past ~10k entries.
        selected = self._selected()
        self._shown = filter_entries(self._entries, self._query.get())
        self._tree.delete(*self._tree.get_children())
        for position, entry in enumerate(self._shown):
            self._tree.insert(
                "", "end", iid=str(position),
                values=(
                    local_time(entry),
                    preview(entry),
                    format_duration(entry.audio_seconds),
                    entry.status,
                ),
                tags=(_ROW_TAGS[entry.status],) if entry.status in _ROW_TAGS else (),
            )
        if selected is not None:
            self._reselect(selected.key)
        self._show_hint(self._hint_text)
        self._render_selection()

    def _reselect(self, key: tuple[str, int]) -> None:
        for position, entry in enumerate(self._shown):
            if entry.key == key:
                self._tree.selection_set(str(position))
                self._tree.see(str(position))
                return

    def _selected(self) -> HistoryEntry | None:
        selection = self._tree.selection()
        if not selection:
            return None
        return self._shown[int(selection[0])]

    # ------------------------------------------------------------------
    # Detail and actions
    # ------------------------------------------------------------------

    def _render_selection(self) -> None:
        entry = self._selected()
        text, note, is_error = self._describe(entry)
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("1.0", text)
        self._detail.configure(state="disabled")
        self._show_status(note, error=is_error)

        has_audio = entry is not None and _audio_exists(entry)
        retrying = entry is not None and self._retries.get(entry.key, "") is None
        _set_enabled(self._copy_button, bool(text.strip()) and not retrying)
        _set_enabled(
            self._play_button, has_audio or self._playback_end is not None,
        )
        _set_enabled(self._retry_button, has_audio and not retrying)
        _set_enabled(self._export_button, has_audio)
        _set_enabled(self._delete_button, entry is not None)
        _set_enabled(self._delete_all_button, bool(self._entries))
        self._retry_button.configure(
            text="Transcribe" if entry is not None and entry.is_cancelled else "Retry",
        )

    def _describe(self, entry: HistoryEntry | None) -> tuple[str, str, bool]:
        """Detail text, status note, and whether the note is an error."""
        if entry is None:
            return "", "", False
        if entry.key in self._retries:
            outcome = self._retries[entry.key]
            if outcome is None:
                return _TRANSCRIBING, "", False
            if isinstance(outcome, Exception):
                return entry.text, f"Transcription failed: {outcome}", True
            return outcome, "New transcription (not saved to history).", False
        if entry.error:
            return entry.text, f"Failed: {entry.error}", True
        if not _audio_exists(entry) and (entry.is_cancelled or not entry.text):
            return entry.text, "No recording was saved for this entry.", False
        return entry.text, "", False

    def _copy(self) -> None:
        text = self._detail.get("1.0", "end-1c")
        if not text.strip():
            return
        self._window.clipboard_clear()
        self._window.clipboard_append(text)
        self._show_status("Copied.")

    def _retry(self) -> None:
        entry = self._selected()
        if entry is None or not _audio_exists(entry):
            return
        key = entry.key
        self._retries[key] = None
        self._render_selection()

        def on_done(outcome: RetryOutcome) -> None:
            self._schedule(lambda: self._retry_finished(key, outcome))

        try:
            self._retranscribe(entry.audio_path, on_done)
        except Exception as exc:  # noqa: BLE001
            self._retry_finished(key, exc)

    def _retry_finished(self, key: tuple[str, int], outcome: RetryOutcome) -> None:
        self._retries[key] = outcome
        if self._window is None:
            return
        selected = self._selected()
        if selected is not None and selected.key == key:
            self._render_selection()

    def _delete(self) -> None:
        entry = self._selected()
        if entry is None:
            return
        if not messagebox.askyesno(
            "Delete dictation",
            "Delete this dictation's text and recording from history?",
            parent=self._window,
        ):
            return
        self._stop()
        self._show_status("Deleting…")
        self._run_in_background(
            lambda: self._report_deleted(
                lambda: delete_entries(self._diagnostics_dir, [entry], utc_now()),
            ),
        )

    def _delete_all(self) -> None:
        if not messagebox.askyesno(
            "Delete all history",
            "Delete every saved dictation's text and recording? "
            "This cannot be undone.",
            parent=self._window,
        ):
            return
        self._stop()
        self._show_status("Deleting…")
        self._run_in_background(
            lambda: self._report_deleted(
                lambda: delete_all_history(self._diagnostics_dir, utc_now()),
            ),
        )

    def _report_deleted(self, delete: Callable[[], int]) -> None:
        try:
            delete()
        except Exception as exc:  # noqa: BLE001 - reported in the window
            _logger.warning("History deletion failed", exc_info=True)
            self._schedule(
                lambda: self._show_status(f"Could not delete: {exc}", error=True),
            )
            return
        self._schedule(self.refresh)

    def _export(self) -> None:
        entry = self._selected()
        if entry is None or not _audio_exists(entry):
            return
        selected = filedialog.asksaveasfilename(
            parent=self._window,
            title="Save audio as",
            initialfile=default_export_name(entry),
            defaultextension=".wav",
            filetypes=(("WAV audio", "*.wav"), ("All files", "*.*")),
        )
        if not selected:
            return
        self._run_in_background(lambda: self._export_to(entry, Path(selected)))

    def _export_to(self, entry: HistoryEntry, destination: Path) -> None:
        try:
            export_audio(self._diagnostics_dir, entry, destination)
        except (OSError, ValueError) as exc:
            self._schedule(
                lambda: self._show_status(f"Could not save the audio: {exc}", error=True),
            )
            return
        self._schedule(lambda: self._show_status(f"Saved {destination.name}."))

    def _toggle_play(self) -> None:
        if self._playback_end is not None:
            self._stop()
            self._render_selection()
            return
        entry = self._selected()
        if entry is None or not _audio_exists(entry):
            return
        try:
            seconds = _play_wav(entry.audio_path)
        except Exception as exc:  # noqa: BLE001
            self._show_status(f"Could not play the recording: {exc}", error=True)
            return
        self._playback_end = self._window.after(
            int(seconds * 1000) + 200, self._playback_finished,
        )
        self._play_button.configure(text="Stop")

    def _playback_finished(self) -> None:
        self._playback_end = None
        self._play_button.configure(text="Play")
        self._render_selection()

    def _stop(self) -> None:
        if self._playback_end is None:
            return
        self._window.after_cancel(self._playback_end)
        self._playback_end = None
        self._play_button.configure(text="Play")
        try:
            _stop_playback()
        except Exception:  # nothing left to stop
            _logger.debug("History playback stop failed", exc_info=True)

    def _show_status(self, message: str, error: bool = False) -> None:
        if self._window is None:
            return
        self._status.configure(
            text=message,
            foreground=(
                self._theme.status_error if error else self._theme.status_neutral
            ),
        )


def _audio_exists(entry: HistoryEntry) -> bool:
    return entry.audio_path is not None and entry.audio_path.is_file()


def _set_enabled(button: ttk.Button, enabled: bool) -> None:
    button.state(["!disabled"] if enabled else ["disabled"])
