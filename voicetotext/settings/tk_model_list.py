import tkinter
from collections.abc import Callable
from tkinter import ttk

from voicetotext.settings.download_progress import DownloadProgressSnapshot
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.service import SettingsService
from voicetotext.settings.signal import Signal
from voicetotext.settings.theme import CONTROL_GAP
from voicetotext.settings.tk_model_row import TkModelRow
from voicetotext.settings.transcription_mode import TranscriptionMode

ModelAction = Callable[[str], None]


def _ignore(_name: str) -> None:
    pass


class TkModelList(ttk.Frame):
    """The model catalog as a stack of :class:`TkModelRow` cards.

    A plain controller that owns the row widgets, the per-name download
    snapshots, and the selection. The page it sits on scrolls, so the
    list itself does not.
    """

    def __init__(
        self,
        parent: tkinter.Misc,
        service: SettingsService,
        *,
        on_download: ModelAction = _ignore,
        on_cancel: ModelAction = _ignore,
        on_delete: ModelAction = _ignore,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._on_download = on_download
        self._on_cancel = on_cancel
        self._on_delete = on_delete
        self._mode: TranscriptionMode | None = None
        self._views: list[ModelView] = []
        self._rows: dict[str, TkModelRow] = {}
        self._downloads: dict[str, DownloadProgressSnapshot] = {}
        self._selected = ""
        self.selection_changed = Signal()
        self._rebuild()

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        for row in self._rows.values():
            row.destroy()
        self._rows.clear()
        self._views = list(self._service.list_models(self._mode))
        self._selected = self._service.selected_model()
        for view in self._views:
            self._add_row(view)
        self._apply_selection()

    def _add_row(self, view: ModelView) -> None:
        row = TkModelRow(self, view)
        row.download_requested.connect(self._on_download)
        row.cancel_requested.connect(self._on_cancel)
        row.delete_requested.connect(self._on_delete)
        row.pack(fill="x", pady=(0, CONTROL_GAP))
        self._bind_row_click(row, view.name)
        self._rows[view.name] = row
        snapshot = self._downloads.get(view.name)
        if snapshot is not None:
            row.update_download(snapshot)

    def _bind_row_click(self, row: TkModelRow, name: str) -> None:
        targets = (
            row,
            row.name_label(),
            row.size_label(),
            row.description_label(),
        )
        for widget in targets:
            widget.bind(
                "<Button-1>", lambda _event, n=name: self.select_model(n),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_mode(self, mode: TranscriptionMode) -> None:
        self._mode = mode
        self._rebuild()

    def selected_model_name(self) -> str:
        return self._selected

    def set_selected(self, name: str) -> None:
        if name == self._selected:
            return
        self._selected = name
        self._apply_selection()

    def select_model(self, name: str) -> None:
        view = self._view_for(name)
        if view is None or not view.installed:
            return
        self.set_selected(name)
        self.selection_changed.emit(name)

    def set_download_progress(
        self, name: str, snapshot: DownloadProgressSnapshot,
    ) -> None:
        self._downloads[name] = snapshot
        row = self._rows.get(name)
        if row is not None:
            row.update_download(snapshot)

    def download_snapshot(self, name: str) -> DownloadProgressSnapshot | None:
        return self._downloads.get(name)

    def clear_download(self, name: str) -> None:
        self._downloads.pop(name, None)
        row = self._rows.get(name)
        if row is not None:
            row.update_download(None)

    def update_install_state(self, name: str) -> None:
        self._rebuild()

    def row_for(self, name: str) -> TkModelRow | None:
        return self._rows.get(name)

    def row_names(self) -> list[str]:
        return [view.name for view in self._views]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _view_for(self, name: str) -> ModelView | None:
        for view in self._views:
            if view.name == name:
                return view
        return None

    def _apply_selection(self) -> None:
        for name, row in self._rows.items():
            row.set_selected(name == self._selected)
