from PySide6.QtCore import QModelIndex, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QListView,
    QVBoxLayout,
    QWidget,
)

from voicetotext.settings.qt_model_list_model import QtModelListModel
from voicetotext.settings.service import SettingsService


_SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "ja", "ko", "zh",
]


class TranscriptionPage(QWidget):
    """Page for picking the active Whisper model, language, device, and engine.

    Uses a :class:`QListView` + :class:`QtModelListModel` so the model
    catalog renders as native item views (deferred to a delegate in the
    main window). Exposes imperative setters/getters so the enclosing
    window can build a :class:`SettingsDraft` on Save.

    The Engine combo is hardware-aware: it asks the service for engine
    candidates compatible with the *currently selected* model on the
    detected hardware, and re-populates whenever the model changes.
    The recommended engine is preselected with a reason label.
    """

    def __init__(
        self, service: SettingsService, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._build()

    def _build(self) -> None:
        draft = self._service.current_draft()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        layout.addWidget(self._make_heading("Models"))
        self._list_model = QtModelListModel(self._service, self)
        self._list_view = QListView()
        self._list_view.setObjectName("model-list")
        self._list_view.setModel(self._list_model)
        self._list_view.setSelectionMode(QListView.SingleSelection)
        self._list_view.clicked.connect(self._on_model_clicked)
        layout.addWidget(self._list_view, stretch=1)
        self._select_initial_model(draft.model_size)

        layout.addWidget(self._make_heading("Settings"))
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self._language_combo = QComboBox()
        self._language_combo.addItems(_SUPPORTED_LANGUAGES)
        self._language_combo.setCurrentText(draft.language)
        form.addRow("Language:", self._language_combo)

        self._device_combo = QComboBox()
        self._device_combo.addItems(self._service.available_devices())
        self._device_combo.setCurrentText(draft.device)
        form.addRow("Device:", self._device_combo)

        self._engine_combo = QComboBox()
        self._engine_combo.setObjectName("engine-combo")
        self._engine_reason_label = QLabel("")
        self._engine_reason_label.setObjectName("engine-reason")
        form.addRow("Engine:", self._engine_combo)
        form.addRow("", self._engine_reason_label)
        self._populate_engine_combo(draft.model_size, current_engine=draft.engine)

        layout.addLayout(form)

    def _make_heading(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("page-heading")
        return label

    def _select_initial_model(self, name: str) -> None:
        self._list_model.set_selected(name)
        self._sync_view_selection(name)

    def _on_model_clicked(self, index: QModelIndex) -> None:
        view = self._list_model.data(index, QtModelListModel.MODEL_ROLE)
        if view is None or not view.installed:
            return
        self._list_model.set_selected(view.name)
        self._populate_engine_combo(view.name)

    def _sync_view_selection(self, name: str) -> None:
        index = self._list_model.index_for_name(name)
        if index.isValid():
            self._list_view.setCurrentIndex(index)

    def _populate_engine_combo(
        self, model_name: str, current_engine: str = "auto",
    ) -> None:
        candidates = tuple(self._service.engine_candidates_for(model_name) or ())
        self._engine_combo.clear()
        self._engine_combo.addItem("auto", userData="auto")
        for candidate in candidates:
            label = candidate.display_name
            if candidate.is_recommended:
                label += "  (recommended)"
            self._engine_combo.addItem(label, userData=candidate.engine_id)
        self._select_engine_combo_value(current_engine)
        reason = candidates[0].reason if candidates else ""
        self._engine_reason_label.setText(str(reason))

    def _select_engine_combo_value(self, engine_id: str) -> None:
        for idx in range(self._engine_combo.count()):
            if self._engine_combo.itemData(idx) == engine_id:
                self._engine_combo.setCurrentIndex(idx)
                return
        self._engine_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Public getters/setters (consumed by the main window)
    # ------------------------------------------------------------------

    def list_model(self) -> QtModelListModel:
        return self._list_model

    def list_view(self) -> QListView:
        return self._list_view

    def current_model(self) -> str:
        return self._list_model.selected_model_name()

    def set_model(self, name: str) -> None:
        self._list_model.set_selected(name)
        self._sync_view_selection(name)

    def current_language(self) -> str:
        return self._language_combo.currentText()

    def set_language(self, language: str) -> None:
        self._language_combo.setCurrentText(language)

    def current_device(self) -> str:
        return self._device_combo.currentText()

    def set_device(self, device: str) -> None:
        self._device_combo.setCurrentText(device)

    def current_engine(self) -> str:
        data = self._engine_combo.currentData()
        return str(data) if data is not None else "auto"

    def set_engine(self, engine: str) -> None:
        self._select_engine_combo_value(engine)

    # ------------------------------------------------------------------
    # Download/delete commands (proxy to the service, tolerate errors
    # the same way the CustomTkinter tab did)
    # ------------------------------------------------------------------

    def request_download(self, name: str) -> None:
        self._service.start_download(name)

    def request_pause(self, name: str) -> None:
        try:
            self._service.pause_download(name)
        except LookupError:
            pass

    def request_resume(self, name: str) -> None:
        self._service.resume_download(name)

    def request_cancel(self, name: str) -> None:
        try:
            self._service.cancel_download(name)
        except LookupError:
            pass

    def request_delete(self, name: str) -> None:
        try:
            self._service.delete_model(name)
        except ValueError:
            pass
