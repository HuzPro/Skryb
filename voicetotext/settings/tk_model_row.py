import tkinter
from tkinter import ttk

from voicetotext.settings.download_progress import (
    DownloadProgressSnapshot,
    percent_for,
)
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.signal import Signal
from voicetotext.settings.theme import (
    CARD_BODY_STYLE,
    CARD_MUTED_STYLE,
    CARD_PADDING,
    CARD_STYLE,
    CARD_TITLE_STYLE,
    SMALL_BUTTON_STYLE,
    THIN_PROGRESS_STYLE,
    is_card_selected,
    set_card_selected,
)


_DOWNLOADING_STATES = frozenset({"queued", "downloading", "paused"})
_DESCRIPTION_WRAP_PX = 430
_INSTALLED_BADGE = "✓ Installed"


class TkModelRow(ttk.Frame):
    """One model as a card in the transcription page's model list.

    Owns the action button (Download / Cancel / Delete depending on state)
    and the thin progress bar (shown only while a download is active).
    Emits one of ``download_requested`` / ``cancel_requested`` /
    ``delete_requested`` (callback-list :class:`Signal`) when the button is
    clicked, keyed on the model's name.

    The row owns no domain logic: callers feed it fresh :class:`ModelView`
    snapshots via :meth:`update_view` and :class:`DownloadProgressSnapshot`
    (or ``None``) via :meth:`update_download`.
    """

    def __init__(self, parent: tkinter.Misc, view: ModelView) -> None:
        super().__init__(parent, style=CARD_STYLE, padding=CARD_PADDING)
        self._name = view.name
        self._installed = view.installed
        self._description = view.description
        self._download_active = False
        self._progress_shown = False
        self.download_requested = Signal()
        self.cancel_requested = Signal()
        self.delete_requested = Signal()
        self._build()
        self.update_view(view)

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self._header = ttk.Frame(self, style=CARD_BODY_STYLE)
        self._header.grid(row=0, column=0, sticky="ew")

        self._name_label = ttk.Label(self._header, text=self._name, style=CARD_TITLE_STYLE)
        self._name_label.pack(side="left")
        self._size_label = ttk.Label(self._header, text="", style=CARD_MUTED_STYLE)
        self._size_label.pack(side="left", padx=(8, 0))
        self._badge = ttk.Label(self._header, text=_INSTALLED_BADGE, style="CardSuccess.TLabel")

        self._description_label = ttk.Label(
            self, text="", style=CARD_MUTED_STYLE,
            wraplength=_DESCRIPTION_WRAP_PX, justify="left",
        )
        self._description_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self._action_button = ttk.Button(
            self, text="", width=9, style=SMALL_BUTTON_STYLE,
            command=self._on_action_clicked,
        )
        self._action_button.grid(row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))

        self._progress = ttk.Progressbar(
            self, maximum=100, mode="determinate", style=THIN_PROGRESS_STYLE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def model_name(self) -> str:
        return self._name

    def update_view(self, view: ModelView) -> None:
        self._installed = view.installed
        self._description = view.description
        self._size_label.configure(text=view.display_size)
        self._description_label.configure(text=view.description)
        if view.installed:
            self._badge.pack(side="left", padx=(10, 0))
        else:
            self._badge.pack_forget()
        self.configure(cursor="hand2" if view.installed else "")
        self._refresh_action_button()

    def update_download(self, snapshot: DownloadProgressSnapshot | None) -> None:
        if snapshot is None:
            self._show_idle()
            return
        if snapshot.state == "failed":
            self._show_failed(snapshot)
            return
        self._show_progress_for(snapshot)

    def set_selected(self, selected: bool) -> None:
        """Tint the whole card and outline it in the accent colour."""
        set_card_selected(self, selected)

    def is_selected(self) -> bool:
        return is_card_selected(self)

    # ------------------------------------------------------------------
    # Test accessors
    # ------------------------------------------------------------------

    def name_label(self) -> ttk.Label:
        return self._name_label

    def size_label(self) -> ttk.Label:
        return self._size_label

    def description_label(self) -> ttk.Label:
        return self._description_label

    def progress_bar(self) -> ttk.Progressbar:
        return self._progress

    def action_button(self) -> ttk.Button:
        return self._action_button

    def progress_visible(self) -> bool:
        return self._progress_shown

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _show_idle(self) -> None:
        self._download_active = False
        self._hide_progress()
        self._description_label.configure(text=self._description)
        self._refresh_action_button()

    def _show_failed(self, snapshot: DownloadProgressSnapshot) -> None:
        self._download_active = False
        self._hide_progress()
        error = snapshot.error_message or "download failed"
        self._description_label.configure(text=f"{self._description}: {error}")
        self._refresh_action_button()

    def _show_progress_for(self, snapshot: DownloadProgressSnapshot) -> None:
        self._download_active = snapshot.state in _DOWNLOADING_STATES
        if self._download_active:
            self._show_progress()
            self._progress.configure(value=percent_for(snapshot))
        else:
            self._hide_progress()
        self._description_label.configure(text=self._description)
        self._refresh_action_button()

    def _show_progress(self) -> None:
        if not self._progress_shown:
            self._progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
            self._progress_shown = True

    def _hide_progress(self) -> None:
        if self._progress_shown:
            self._progress.grid_remove()
            self._progress_shown = False

    def _refresh_action_button(self) -> None:
        self._action_button.configure(text=self._action_label())

    def _action_label(self) -> str:
        if self._download_active:
            return "Cancel"
        if self._installed:
            return "Delete"
        return "Download"

    def _on_action_clicked(self) -> None:
        if self._download_active:
            self.cancel_requested.emit(self._name)
        elif self._installed:
            self.delete_requested.emit(self._name)
        else:
            self.download_requested.emit(self._name)
