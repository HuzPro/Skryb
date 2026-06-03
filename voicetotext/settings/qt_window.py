from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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
from voicetotext.settings.qt_download_bridge import QtDownloadBridge
from voicetotext.settings.qt_engine_bridge import QtEngineBridge
from voicetotext.settings.qt_hotkey_capture import CaptureBridge
from voicetotext.settings.qt_model_list_model import DownloadProgressSnapshot
from voicetotext.settings.qt_pages.hotkeys_page import HotkeysPage
from voicetotext.settings.qt_pages.mode_page import ModePage
from voicetotext.settings.qt_pages.output_page import OutputPage
from voicetotext.settings.qt_pages.transcription_page import TranscriptionPage
from voicetotext.settings.qt_state_bridge import QtStateBridge
from voicetotext.settings.service import SettingsService
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft


_STATUS_ERROR_STYLE = "color: #FF6666;"
_STATUS_NEUTRAL_STYLE = "color: #CCCCCC;"


class QtSettingsWindow(QMainWindow):
    """Top-level Qt settings window — sidebar nav + stacked pages + footer.

    The window depends only on :class:`SettingsService` — it never imports
    domain types directly, so the view layer stays decoupled from
    transcription, downloads, and config internals.

    Engine and state-machine events arrive on the event bus from any
    thread (the hot-swap worker for engine events, the recording/
    transcription worker for state transitions). They are re-emitted
    through :class:`QtEngineBridge` and :class:`QtStateBridge` — both
    are ``QObject`` children of this window with GUI-thread affinity,
    so ``Qt.AutoConnection`` automatically queues the slot invocation
    onto the GUI event loop. Tests may invoke the slots directly on
    the GUI thread to bypass the bridge.
    """

    def __init__(
        self,
        service: SettingsService,
        first_run: bool = False,
        on_save: Callable[[ChangeSet], None] | None = None,
        on_capture_state_changed: Callable[[bool], None] | None = None,
        capture_bridge: CaptureBridge | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Ensure ``close()`` triggers ``deleteLater`` *after* ``closeEvent``
        # runs — without this, QMainWindow only hides on close(), leaving
        # the window alive in memory and its bridges subscribed to the
        # service. A later Python-GC reaping would destroy the window
        # without firing closeEvent, leaking subscriptions and crashing
        # the next fanout with "Signal source has been deleted".
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._service = service
        self._first_run = first_run
        self._on_save = on_save
        self._on_capture_state_changed = on_capture_state_changed
        self._capture_bridge = capture_bridge
        self._pending_changes: ChangeSet | None = None

        self._engine_bridge = QtEngineBridge(parent=self)
        self._state_bridge = QtStateBridge(parent=self)
        self._download_bridge = QtDownloadBridge(parent=self)
        self._engine_bridge.load_started.connect(self.on_engine_load_started)
        self._engine_bridge.load_completed.connect(self.on_engine_load_completed)
        self._engine_bridge.load_failed.connect(self.on_engine_load_failed)
        self._state_bridge.state_changed.connect(self.on_state_changed)
        self._download_bridge.progress.connect(self.on_download_progress)
        self._download_bridge.state.connect(self.on_download_state)
        self._download_bridge.completed.connect(self.on_download_completed)
        self._download_bridge.failed.connect(self.on_download_failed)
        self._download_bridge.deleted.connect(self.on_model_deleted)

        self.setWindowTitle(
            "VoiceToText Setup" if first_run else "VoiceToText Settings"
        )
        self.resize(720, 600)

        self._build_ui()
        self._service.subscribe_engine(self._engine_bridge)
        self._service.subscribe_state_changes(self._state_bridge)
        self._service.subscribe_downloads(self._download_bridge)
        self._service.subscribe_model_deleted(self._download_bridge.on_model_deleted)
        self._refresh_save_enabled()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self._sidebar = self._build_sidebar()
        body_layout.addWidget(self._sidebar)

        self._pages = QStackedWidget()
        self._transcription_page = TranscriptionPage(self._service)
        self._transcription_page.download_error.connect(self._on_download_error)
        self._transcription_page.delete_error.connect(self._on_delete_error)
        self._mode_page = ModePage()
        self._mode_page.mode_changed.connect(self._transcription_page.apply_mode)
        self._hotkeys_page = HotkeysPage(capture_bridge=self._capture_bridge)
        self._hotkeys_page.recording_state_changed.connect(
            self._on_recording_state_changed
        )
        self._output_page = OutputPage()
        self._pages.addWidget(self._transcription_page)
        self._pages.addWidget(self._mode_page)
        self._pages.addWidget(self._hotkeys_page)
        self._pages.addWidget(self._output_page)
        body_layout.addWidget(self._pages, stretch=1)

        self._apply_initial_draft()

        root_layout.addWidget(body, stretch=1)
        root_layout.addWidget(self._build_footer())

        self._sidebar.setCurrentRow(0)

    def _build_sidebar(self) -> QListWidget:
        sidebar = QListWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(168)
        sidebar.addItems(["Transcription", "Mode", "Hotkeys", "Output"])
        sidebar.currentRowChanged.connect(self._on_sidebar_changed)
        return sidebar

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setObjectName("footer")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(16, 8, 16, 12)

        self._status_label = QLabel("")
        self._status_label.setObjectName("status")
        self._status_label.setStyleSheet(_STATUS_NEUTRAL_STYLE)
        layout.addWidget(self._status_label, stretch=1)

        button_text = "Save & Start" if self._first_run else "Save & Apply"
        self._save_button = QPushButton(button_text)
        self._save_button.setObjectName("save-button")
        self._save_button.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_button)

        return footer

    def _apply_initial_draft(self) -> None:
        draft = self._service.current_draft()
        self._hotkeys_page.set_combination(draft.hotkey_combination)
        self._hotkeys_page.set_mode(draft.hotkey_mode)
        self._output_page.set_mode(draft.output_mode)
        self._mode_page.set_mode(draft.transcription_mode)

    def _on_sidebar_changed(self, row: int) -> None:
        if 0 <= row < self._pages.count():
            self._pages.setCurrentIndex(row)

    # ------------------------------------------------------------------
    # Save flow
    # ------------------------------------------------------------------

    def collect_draft(self) -> SettingsDraft:
        return SettingsDraft(
            model_size=self._transcription_page.current_model(),
            language=self._transcription_page.current_language(),
            device=self._transcription_page.current_device(),
            engine=self._transcription_page.current_engine(),
            hotkey_combination=self._hotkeys_page.current_combination(),
            hotkey_mode=self._hotkeys_page.current_mode(),
            output_mode=self._output_page.current_mode(),
            transcription_mode=self._mode_page.current_mode(),
        )

    def _on_save_clicked(self) -> None:
        if self._first_run and not self._service.any_model_installed():
            self._show_status(
                "Please download at least one model before saving.",
                error=True,
            )
            return

        draft = self.collect_draft()
        if not self._service.is_installed(draft.model_size):
            self._show_status(
                f"'{draft.model_size}' is not installed. Download it first.",
                error=True,
            )
            return
        changes = self._service.apply_draft(draft)
        self._pending_changes = changes

        if changes.engine_reload_required and not self._first_run:
            self._disable_save_button()
            self._show_status("Loading model — please wait…")
            self._service.trigger_engine_swap()
            return

        self._finalize_save()

    def _finalize_save(self) -> None:
        changes = self._pending_changes
        self._pending_changes = None
        if self._on_save is not None and changes is not None:
            self._on_save(changes)
        self.close()

    def _disable_save_button(self) -> None:
        self._save_button.setEnabled(False)

    def _refresh_save_enabled(self) -> None:
        allowed = (
            self._service.can_apply_now()
            and self._pending_changes is None
        )
        self._save_button.setEnabled(allowed)

    def _show_status(self, message: str, error: bool = False) -> None:
        self._status_label.setText(message)
        self._status_label.setStyleSheet(
            _STATUS_ERROR_STYLE if error else _STATUS_NEUTRAL_STYLE
        )

    # ------------------------------------------------------------------
    # Engine slots — invoked on the GUI thread via QtEngineBridge's
    # queued signals (AutoConnection queues across threads).
    # ------------------------------------------------------------------

    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None:
        self._show_status(f"Loading '{event.model_name}'…")

    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        self._save_button.setEnabled(True)
        self._show_status(f"Loaded '{event.model_name}'.")
        self._finalize_save()
        self._refresh_save_enabled()

    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None:
        self._save_button.setEnabled(True)
        self._pending_changes = None
        self._show_status(
            f"Failed to load '{event.model_name}': {event.error}", error=True
        )
        self._refresh_save_enabled()

    # ------------------------------------------------------------------
    # Download slots — invoked on the GUI thread via QtDownloadBridge.
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
        self._transcription_page.list_model().set_download_progress(
            event.job_id, snapshot
        )

    def on_download_state(self, event: DownloadStateChangedEvent) -> None:
        # Reflect non-progress lifecycle states (queued, paused) onto the
        # row so the action button can flip between Cancel/Resume.
        list_model = self._transcription_page.list_model()
        existing = list_model.data(
            list_model.index_for_name(event.job_id),
            list_model.DOWNLOAD_ROLE,
        )
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
        list_model.set_download_progress(event.job_id, snapshot)

    def on_download_completed(self, event: DownloadCompletedEvent) -> None:
        list_model = self._transcription_page.list_model()
        list_model.clear_download(event.job_id)
        list_model.update_install_state(event.job_id)

    def on_download_failed(self, event: DownloadFailedEvent) -> None:
        snapshot = DownloadProgressSnapshot(
            state="failed",
            bytes_done=0,
            bytes_total=0,
            speed_bps=0.0,
            eta_seconds=0.0,
            error_message=event.error,
        )
        self._transcription_page.list_model().set_download_progress(
            event.job_id, snapshot
        )
        self._show_status(
            f"Download '{event.job_id}' failed: {event.error}", error=True,
        )

    def _on_download_error(self, name: str, error: str) -> None:
        """Surface synchronous download errors (e.g., HF resolve failure)."""
        self._show_status(
            f"Could not start download of '{name}': {error}", error=True,
        )

    def _on_delete_error(self, name: str, error: str) -> None:
        """Surface synchronous delete errors (e.g., active-model guard)."""
        self._show_status(
            f"Cannot delete '{name}': {error}", error=True,
        )

    def on_model_deleted(self, event: ModelDeletedEvent) -> None:
        """Refresh the row of the just-deleted model.

        Drops any stale download snapshot and asks the list model to
        re-query the service for the row's install state, which flips
        the action button from "Delete" back to "Download".
        """
        list_model = self._transcription_page.list_model()
        list_model.clear_download(event.name)
        list_model.update_install_state(event.name)

    # ------------------------------------------------------------------
    # State-machine listener — enables/disables Save while recording.
    # ------------------------------------------------------------------

    def on_state_changed(self, event: StateChangedEvent) -> None:
        self._refresh_save_enabled()

    def _on_recording_state_changed(self, recording: bool) -> None:
        if self._on_capture_state_changed is not None:
            self._on_capture_state_changed(recording)

    # ------------------------------------------------------------------
    # Close/teardown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        # Unsubscribe from the service before calling ``super().closeEvent``
        # so no fanout call can reach the bridges after we're closing.
        # Any in-flight emit that slipped through queues to the bridges'
        # connected slots — Qt drops queued invocations to destroyed
        # QObjects, so this is safe.
        self._service.unsubscribe_engine(self._engine_bridge)
        self._service.unsubscribe_state_changes(self._state_bridge)
        self._service.unsubscribe_downloads(self._download_bridge)
        self._service.unsubscribe_model_deleted(
            self._download_bridge.on_model_deleted,
        )
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Test accessors
    # ------------------------------------------------------------------

    def transcription_page(self) -> TranscriptionPage:
        return self._transcription_page

    def mode_page(self) -> ModePage:
        return self._mode_page

    def hotkeys_page(self) -> HotkeysPage:
        return self._hotkeys_page

    def output_page(self) -> OutputPage:
        return self._output_page

    def save_button(self) -> QPushButton:
        return self._save_button

    def status_label(self) -> QLabel:
        return self._status_label
