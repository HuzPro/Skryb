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


def _should_show_scrollbar(content_height: int, viewport_height: int) -> bool:
    """The scrollbar is needed only when the rows overflow the viewport."""
    return content_height > viewport_height


def _scroll_region_height(content_height: int, viewport_height: int) -> int:
    """Scroll region never shorter than the viewport.

    A region shorter than the viewport lets the canvas scroll into empty
    space above the content (dragging the thumb up pushes the rows down).
    Clamping to the viewport height makes "nothing to scroll" actually
    mean nothing moves.
    """
    return max(content_height, viewport_height)


class TkModelList(ttk.Frame):
    """Scrollable model catalog: a ``Canvas`` of per-model :class:`TkModelRow`.

    A plain controller that owns the row widgets, the per-name download
    snapshots, and the selection. The vertical scrollbar auto-hides when
    the rows fit, and relayout work is coalesced through ``after_idle`` so
    dragging the window edge does not thrash the canvas.
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
        self._section_labels: list[ttk.Label] = []
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
        self._scrollbar = ttk.Scrollbar(
            self, orient="vertical", command=self._canvas.yview,
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        # The scrollbar is packed on demand by ``_sync_scroll``.
        self._scrollbar_visible = False
        self._last_width = -1
        self._content_height = 0
        self._last_region_height: int | None = None

        self._inner = ttk.Frame(self._canvas)
        self._inner_id = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw",
        )
        self._inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, event: tkinter.Event) -> None:
        # Content height changes only when rows are added/removed or a
        # progress bar shows/hides. Read it from the event (cheap) rather
        # than ``winfo_reqheight`` (a full geometry recompute that caused
        # the resize stutter).
        self._content_height = event.height
        self._sync_scroll(self._canvas.winfo_height())

    def _on_canvas_configure(self, event: tkinter.Event) -> None:
        # Match the inner frame's width to the viewport only when the width
        # actually changes; a height-only resize then does no reflow at all.
        if event.width != self._last_width:
            self._canvas.itemconfigure(self._inner_id, width=event.width)
            self._last_width = event.width
        self._sync_scroll(event.height)

    def _sync_scroll(self, viewport_height: int) -> None:
        region_height = _scroll_region_height(
            self._content_height, viewport_height,
        )
        if region_height != self._last_region_height:
            self._canvas.configure(scrollregion=(0, 0, 0, region_height))
            self._last_region_height = region_height
        self._set_scrollbar_visible(
            _should_show_scrollbar(self._content_height, viewport_height),
        )

    def _refresh_content_height(self) -> None:
        # One-off measure after a rebuild (a content change, never during a
        # resize), so the scrollbar state is correct before the first
        # ``<Configure>`` lands.
        if not self._canvas.winfo_exists():
            return
        self._content_height = self._inner.winfo_reqheight()
        self._sync_scroll(self._canvas.winfo_height())

    def _set_scrollbar_visible(self, visible: bool) -> None:
        if visible == self._scrollbar_visible:
            return
        self._scrollbar_visible = visible
        if visible:
            self._scrollbar.pack(side="right", fill="y")
        else:
            self._scrollbar.pack_forget()
            self._canvas.yview_moveto(0)

    def _on_mousewheel(self, event: tkinter.Event) -> None:
        if self._scrollbar_visible:
            self._canvas.yview_scroll(-int(event.delta / _WHEEL_NOTCH), "units")

    # ------------------------------------------------------------------
    # Row construction
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        for row in self._rows.values():
            row.destroy()
        for label in self._section_labels:
            label.destroy()
        self._rows.clear()
        self._section_labels.clear()
        self._views = list(self._service.list_models(self._mode))
        self._selected = self._service.selected_model()
        legacy_heading_added = False
        for view in self._views:
            if view.legacy and not legacy_heading_added:
                heading = ttk.Label(self._inner, text="Legacy (installed)")
                heading.pack(fill="x", padx=4, pady=(8, 6))
                self._section_labels.append(heading)
                legacy_heading_added = True
            self._add_row(view)
        self._apply_selection()
        self._refresh_content_height()

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
        self._apply_selection()

    def select_model(self, name: str) -> None:
        view = self._view_for(name)
        if view is None or not view.installed or not view.selectable:
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
