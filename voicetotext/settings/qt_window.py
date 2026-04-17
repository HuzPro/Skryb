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
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
    StateChangedEvent,
)
from voicetotext.settings.qt_pages.hotkeys_page import HotkeysPage
from voicetotext.settings.qt_pages.output_page import OutputPage
from voicetotext.settings.qt_pages.transcription_page import TranscriptionPage
from voicetotext.settings.service import (
    EngineEventListener,
    SettingsService,
    StateChangeListener,
)
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft


_STATUS_ERROR_STYLE = "color: #FF6666;"
_STATUS_NEUTRAL_STYLE = "color: #CCCCCC;"


class QtSettingsWindow(QMainWindow):
    """Top-level Qt settings window — sidebar nav + stacked pages + footer.

    The window depends only on :class:`SettingsService` — it never imports
    domain types directly, so the view layer stays decoupled from
    transcription, downloads, and config internals.

    Engine and state-machine events arrive on listener callbacks. They
    may be invoked from any thread when fired via the real event bus;
    in the Qt app they are re-emitted through :class:`QtEngineBridge`
    which marshals to the GUI thread before calling back here. Tests
    invoke the callbacks directly on the GUI thread.
    """

    def __init__(
        self,
        service: SettingsService,
        first_run: bool = False,
        on_save: Callable[[ChangeSet], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._first_run = first_run
        self._on_save = on_save
        self._pending_changes: ChangeSet | None = None

        self._engine_listener = _EngineListenerProxy(self)
        self._state_listener = _StateListenerProxy(self)

        self.setWindowTitle(
            "VoiceToText Setup" if first_run else "VoiceToText Settings"
        )
        self.resize(720, 600)

        self._build_ui()
        self._service.subscribe_engine(self._engine_listener)
        self._service.subscribe_state_changes(self._state_listener)
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
        self._hotkeys_page = HotkeysPage()
        self._output_page = OutputPage()
        self._pages.addWidget(self._transcription_page)
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
        sidebar.addItems(["Transcription", "Hotkeys", "Output"])
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
            hotkey_combination=self._hotkeys_page.current_combination(),
            hotkey_mode=self._hotkeys_page.current_mode(),
            output_mode=self._output_page.current_mode(),
        )

    def _on_save_clicked(self) -> None:
        if self._first_run and not self._service.any_model_installed():
            self._show_status(
                "Please download at least one model before saving.",
                error=True,
            )
            return

        draft = self.collect_draft()
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
    # Engine listener callbacks — invoked on the GUI thread via
    # QtEngineBridge's queued signals.
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
    # State-machine listener — enables/disables Save while recording.
    # ------------------------------------------------------------------

    def on_state_changed(self, event: StateChangedEvent) -> None:
        self._refresh_save_enabled()

    # ------------------------------------------------------------------
    # Close/teardown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._service.unsubscribe_engine(self._engine_listener)
        self._service.unsubscribe_state_changes(self._state_listener)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Test accessors
    # ------------------------------------------------------------------

    def transcription_page(self) -> TranscriptionPage:
        return self._transcription_page

    def hotkeys_page(self) -> HotkeysPage:
        return self._hotkeys_page

    def output_page(self) -> OutputPage:
        return self._output_page

    def save_button(self) -> QPushButton:
        return self._save_button

    def status_label(self) -> QLabel:
        return self._status_label


class _EngineListenerProxy(EngineEventListener):
    """Forwards engine-load events to the owning window.

    A thin indirection keeps :class:`QtSettingsWindow` from having to
    inherit :class:`EngineEventListener` directly (which would drag the
    ABC metaclass into Qt's ``QObject`` hierarchy and conflict with
    shiboken's metaclass).
    """

    def __init__(self, window: QtSettingsWindow) -> None:
        self._window = window

    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None:
        self._window.on_engine_load_started(event)

    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        self._window.on_engine_load_completed(event)

    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None:
        self._window.on_engine_load_failed(event)


class _StateListenerProxy(StateChangeListener):
    """Forwards state-machine transitions to the owning window."""

    def __init__(self, window: QtSettingsWindow) -> None:
        self._window = window

    def on_state_changed(self, event: StateChangedEvent) -> None:
        self._window.on_state_changed(event)
