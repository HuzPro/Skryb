from PySide6.QtCore import QAbstractListModel, QModelIndex, QSize, Qt

from voicetotext.settings.download_progress import DownloadProgressSnapshot
from voicetotext.settings.model_view import ModelView
from voicetotext.settings.service import SettingsService
from voicetotext.settings.transcription_mode import TranscriptionMode


_ROW_HEIGHT = 72

# Re-exported for callers that still import the snapshot from this module
# (`qt_model_row`, `qt_window`). The canonical definition is Qt-free in
# ``download_progress`` so the Tk views can share it.
__all__ = ["DownloadProgressSnapshot", "QtModelListModel"]


class QtModelListModel(QAbstractListModel):
    """List model backing the transcription-page model view.

    Exposes one row per :class:`ModelView` from :class:`SettingsService`.
    Carries three custom roles: the :class:`ModelView` itself, a
    selection flag, and an optional :class:`DownloadProgressSnapshot`
    that the delegate renders as an inline progress bar.
    """

    MODEL_ROLE = Qt.UserRole + 1
    SELECTED_ROLE = Qt.UserRole + 2
    DOWNLOAD_ROLE = Qt.UserRole + 3

    def __init__(
        self, service: SettingsService, parent: QAbstractListModel | None = None
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._views: list[ModelView] = []
        self._downloads: dict[str, DownloadProgressSnapshot] = {}
        self._selected: str = ""
        self._mode: TranscriptionMode | None = None
        self.refresh()

    def set_mode(self, mode: TranscriptionMode) -> None:
        """Filter the list to *mode*'s eligible models (preview before save)."""
        self._mode = mode
        self.refresh()

    def refresh(self) -> None:
        self.beginResetModel()
        self._views = list(self._service.list_models(self._mode))
        self._selected = self._service.selected_model()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._views)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:
        if not index.isValid() or not (0 <= index.row() < len(self._views)):
            return None
        view = self._views[index.row()]
        if role == self.MODEL_ROLE:
            return view
        if role == self.SELECTED_ROLE:
            return view.name == self._selected
        if role == self.DOWNLOAD_ROLE:
            return self._downloads.get(view.name)
        if role == Qt.DisplayRole:
            # Suppressed: the per-row :class:`ModelRowWidget` renders
            # everything. Returning the name here would paint over the
            # widget's content.
            return ""
        if role == Qt.SizeHintRole:
            return QSize(0, _ROW_HEIGHT)
        return None

    def selected_model_name(self) -> str:
        return self._selected

    def index_for_name(self, name: str) -> QModelIndex:
        for row, view in enumerate(self._views):
            if view.name == name:
                return self.index(row)
        return QModelIndex()

    def set_selected(self, name: str) -> None:
        if name == self._selected:
            return
        old_name = self._selected
        self._selected = name
        for row, view in enumerate(self._views):
            if view.name in (old_name, name):
                idx = self.index(row)
                self.dataChanged.emit(idx, idx, [self.SELECTED_ROLE])

    def update_install_state(self, name: str) -> None:
        fresh_by_name = {v.name: v for v in self._service.list_models()}
        fresh = fresh_by_name.get(name)
        if fresh is None:
            return
        for row, view in enumerate(self._views):
            if view.name == name:
                self._views[row] = fresh
                idx = self.index(row)
                self.dataChanged.emit(idx, idx, [self.MODEL_ROLE])
                return

    def set_download_progress(
        self, name: str, snapshot: DownloadProgressSnapshot
    ) -> None:
        self._downloads[name] = snapshot
        self._emit_download_changed(name)

    def clear_download(self, name: str) -> None:
        if name in self._downloads:
            self._downloads.pop(name)
            self._emit_download_changed(name)

    def _emit_download_changed(self, name: str) -> None:
        for row, view in enumerate(self._views):
            if view.name == name:
                idx = self.index(row)
                self.dataChanged.emit(idx, idx, [self.DOWNLOAD_ROLE])
                return
