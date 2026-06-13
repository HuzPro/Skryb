import tkinter
from collections.abc import Callable
from tkinter import ttk

from voicetotext.settings.download_progress import DownloadProgressSnapshot
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.service import SettingsService
from voicetotext.settings.signal import Signal
from voicetotext.settings.tk_model_row import TkModelRow
from voicetotext.settings.transcription_mode import TranscriptionMode

ModelAction = Callable[[str], None]

_WHEEL_NOTCH = 120  # Windows delivers wheel deltas in multiples of 120


def _ignore(_name: str) -> None:
    pass


class TkModelList(ttk.Frame):
    """Scrollable model catalog: a ``Canvas`` of per-model :class:`TkModelRow`.

    A plain controller that owns the row widgets, the per-name download
    snapshots, and the selection, replacing Qt's ``QAbstractListModel`` +
    ``setIndexWidget`` plumbing. Action callbacks (download/cancel/delete)
    are injected once and re-bound to every row, including those rebuilt
    on a mode change. :attr:`selection_changed` fires when the user picks
    an installed row.
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
        self._build_scroll_area()
        self._rebuild()

    # ------------------------------------------------------------------
    # Scroll skeleton
    # ------------------------------------------------------------------

    def _build_scroll_area(self) -> None:
        self._canvas = tkinter.Canvas(self, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self._canvas.yview,
        )
        self._canvas.configure(yscrollcommand=scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._inner = ttk.Frame(self._canvas)
        self._inner_id = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw",
        )
        self._inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, _event: tkinter.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tkinter.Event) -> None:
        self._canvas.itemconfigure(self._inner_id, width=event.width)

    def _on_mousewheel(self, event: tkinter.Event) -> None:
        self._canvas.yview_scroll(-int(event.delta / _WHEEL_NOTCH), "units")

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
        self._apply_selection_visual()

    def _add_row(self, view: ModelView) -> None:
        row = TkModelRow(self._inner, view)
        row.download_requested.connect(self._on_download)
        row.cancel_requested.connect(self._on_cancel)
        row.delete_requested.connect(self._on_delete)
        row.pack(fill="x", padx=4, pady=(0, 6))
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
        self._apply_selection_visual()

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
        fresh = {v.name: v for v in self._service.list_models()}.get(name)
        if fresh is None:
            return
        for index, view in enumerate(self._views):
            if view.name == name:
                self._views[index] = fresh
                row = self._rows.get(name)
                if row is not None:
                    row.update_view(fresh)
                return

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

    def _apply_selection_visual(self) -> None:
        for name, row in self._rows.items():
            if name == self._selected:
                row.configure(relief="solid", borderwidth=1)
            else:
                row.configure(relief="flat", borderwidth=0)
