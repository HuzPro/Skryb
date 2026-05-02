from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from voicetotext.settings.model_view import ModelView
from voicetotext.settings.qt_model_list_model import DownloadProgressSnapshot


_DOWNLOADING_STATES = frozenset({"queued", "downloading", "paused"})


class ModelRowWidget(QWidget):
    """Per-row widget for the transcription page's model list.

    Encapsulates the action button (Download / Cancel / Delete depending
    on state) and the inline progress bar (visible only while a download
    is active). Emits one of ``download_requested`` / ``cancel_requested``
    / ``delete_requested`` when the button is clicked, keyed on the
    model's name so the page can route to ``SettingsService``.

    The widget owns no domain logic — callers feed it fresh
    :class:`ModelView` snapshots via :meth:`update_view` whenever the
    install state changes, and :class:`DownloadProgressSnapshot` (or
    ``None``) via :meth:`update_download` for download progress events.
    """

    download_requested = Signal(str)
    cancel_requested = Signal(str)
    delete_requested = Signal(str)
    pause_requested = Signal(str)
    resume_requested = Signal(str)

    def __init__(self, view: ModelView, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = view.name
        self._installed = view.installed
        self._description = view.description
        self._download_active = False
        self._build()
        self.update_view(view)

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self._name_label = QLabel(self._name)
        self._name_label.setObjectName("model-name")
        header.addWidget(self._name_label)

        self._size_label = QLabel("")
        self._size_label.setObjectName("model-size")
        header.addWidget(self._size_label)

        header.addStretch(1)

        self._action_button = QPushButton("")
        self._action_button.setObjectName("model-action")
        self._action_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._action_button.clicked.connect(self._on_action_clicked)
        header.addWidget(self._action_button)

        outer.addLayout(header)

        self._description_label = QLabel("")
        self._description_label.setObjectName("model-description")
        self._description_label.setWordWrap(True)
        outer.addWidget(self._description_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("model-progress")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setVisible(False)
        outer.addWidget(self._progress_bar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def model_name(self) -> str:
        return self._name

    def update_view(self, view: ModelView) -> None:
        self._installed = view.installed
        self._description = view.description
        self._size_label.setText(view.display_size)
        self._description_label.setText(view.description)
        self._refresh_action_button()

    def update_download(self, snapshot: DownloadProgressSnapshot | None) -> None:
        if snapshot is None:
            self._download_active = False
            self._progress_bar.setVisible(False)
            self._description_label.setText(self._description)
            self._refresh_action_button()
            return

        if snapshot.state == "failed":
            self._download_active = False
            self._progress_bar.setVisible(False)
            error = snapshot.error_message or "download failed"
            self._description_label.setText(f"{self._description} — {error}")
            self._refresh_action_button()
            return

        self._download_active = snapshot.state in _DOWNLOADING_STATES
        if self._download_active:
            self._progress_bar.setVisible(True)
            self._progress_bar.setValue(_percent(snapshot))
        else:
            self._progress_bar.setVisible(False)
        self._description_label.setText(self._description)
        self._refresh_action_button()

    # ------------------------------------------------------------------
    # Test accessors
    # ------------------------------------------------------------------

    def name_label(self) -> QLabel:
        return self._name_label

    def size_label(self) -> QLabel:
        return self._size_label

    def description_label(self) -> QLabel:
        return self._description_label

    def progress_bar(self) -> QProgressBar:
        return self._progress_bar

    def action_button(self) -> QPushButton:
        return self._action_button

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _refresh_action_button(self) -> None:
        if self._download_active:
            self._action_button.setText("Cancel")
        elif self._installed:
            self._action_button.setText("Delete")
        else:
            self._action_button.setText("Download")

    def _on_action_clicked(self) -> None:
        if self._download_active:
            self.cancel_requested.emit(self._name)
        elif self._installed:
            self.delete_requested.emit(self._name)
        else:
            self.download_requested.emit(self._name)


def _percent(snapshot: DownloadProgressSnapshot) -> int:
    if snapshot.bytes_total <= 0:
        return 0
    pct = int((snapshot.bytes_done / snapshot.bytes_total) * 100)
    if pct < 0:
        return 0
    if pct > 100:
        return 100
    return pct
