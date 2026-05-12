from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from voicetotext.core.state_machine import AppState
from voicetotext.tray.duration_format import format_duration
from voicetotext.tray.qt_icons import create_qicon


_TITLE_PREFIX = "VoiceToText — "
_ELAPSED_REFRESH_INTERVAL_MS = 1000


def _run_event_loop() -> None:
    """Run the Qt event loop.

    Isolated to a module-level function so tests can monkey-patch it
    and exercise :meth:`QtTrayAdapter.run` without actually entering
    ``QApplication.exec()``.
    """
    app = QApplication.instance()
    if app is not None:
        app.exec()


class QtTrayAdapter(QObject):
    """``QSystemTrayIcon``-backed :class:`voicetotext.tray.app.TrayIcon`.

    Satisfies the ``TrayIcon`` Protocol (``visible``, ``update_state``,
    ``run``, ``stop``) so :class:`SystemTrayApp` can inject it without
    knowing about Qt.

    ``get_state_label`` is invoked each time the context menu is about
    to show, to keep the disabled state-display entry current — Qt's
    ``QAction`` text is static, so the menu item needs a manual refresh
    hook on ``aboutToShow``.
    """

    # Thread-marshaling seam. ``update_state`` may be called from any
    # thread (state transitions are published synchronously by the
    # transcription worker), but ``QSystemTrayIcon`` mutations must run
    # on the main thread. Emitting this signal lets ``Qt.AutoConnection``
    # queue the slot onto the adapter's owner thread — which is the main
    # thread because the factory is invoked during bootstrap.
    _state_update_requested = Signal(object, str)
    _notification_requested = Signal(str, str)
    # Cross-thread seam for the elapsed-time tooltip: the use case
    # transitions state from the worker thread, ``SystemTrayApp``
    # forwards the provider here, and the QTimer that polls it must
    # be created/owned by the GUI thread. ``object`` allows ``None``
    # to flow through, which means "stop tracking".
    _elapsed_provider_requested = Signal(object)

    def __init__(
        self,
        *,
        initial_state: AppState,
        initial_label: str,
        get_state_label: Callable[[], str],
        on_settings_clicked: Callable[[], None] | None,
        on_quit_clicked: Callable[[], None],
        tray_backend: object | None = None,
    ) -> None:
        super().__init__()
        self._icon = create_qicon(initial_state)
        self._title = _TITLE_PREFIX + initial_label
        self._visible = False
        self._get_state_label = get_state_label
        self._on_settings_clicked = on_settings_clicked
        self._on_quit_clicked = on_quit_clicked

        self._tray = tray_backend if tray_backend is not None else QSystemTrayIcon()
        self._tray.setIcon(self._icon)
        self._tray.setToolTip(self._title)

        self._menu = self._build_menu()
        self._tray.setContextMenu(self._menu)

        self._elapsed_provider: Callable[[], float] | None = None
        self._elapsed_timer: QTimer | None = None

        self._state_update_requested.connect(self._apply_state_update)
        self._notification_requested.connect(self._show_notification)
        self._elapsed_provider_requested.connect(self._apply_elapsed_provider)

    def _build_menu(self) -> QMenu:
        menu = QMenu()
        self._state_action = QAction(self._get_state_label(), menu)
        self._state_action.setEnabled(False)
        menu.addAction(self._state_action)
        menu.addSeparator()

        if self._on_settings_clicked is not None:
            settings_action = QAction("Settings…", menu)
            settings_action.triggered.connect(
                lambda _checked=False: self._on_settings_clicked()
            )
            menu.addAction(settings_action)
            menu.addSeparator()

        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(
            lambda _checked=False: self._on_quit_clicked()
        )
        menu.addAction(quit_action)

        menu.aboutToShow.connect(self._refresh_state_label)
        return menu

    def _refresh_state_label(self) -> None:
        self._state_action.setText(self._get_state_label())

    # ------------------------------------------------------------------
    # TrayIcon Protocol — visibility toggle and state updates.
    # ------------------------------------------------------------------

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = value
        if value:
            self._tray.show()
        else:
            self._tray.hide()

    def update_state(self, state: AppState, label: str) -> None:
        self._state_update_requested.emit(state, label)

    def set_elapsed_provider(
        self, provider: Callable[[], float] | None,
    ) -> None:
        self._elapsed_provider_requested.emit(provider)

    def notify(self, title: str, message: str) -> None:
        self._notification_requested.emit(title, message)

    def _show_notification(self, title: str, message: str) -> None:
        self._tray.showMessage(title, message)

    def _apply_state_update(self, state: AppState, label: str) -> None:
        self._icon = create_qicon(state)
        self._title = _TITLE_PREFIX + label
        self._tray.setIcon(self._icon)
        self._tray.setToolTip(self._title)
        self._refresh_state_label()

    def _apply_elapsed_provider(
        self, provider: Callable[[], float] | None,
    ) -> None:
        self._stop_elapsed_timer()
        self._elapsed_provider = provider
        if provider is None:
            self._tray.setToolTip(self._title)
            return
        self._start_elapsed_timer()
        self._refresh_elapsed_tooltip()

    def _start_elapsed_timer(self) -> None:
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(_ELAPSED_REFRESH_INTERVAL_MS)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed_tooltip)
        self._elapsed_timer.start()

    def _stop_elapsed_timer(self) -> None:
        if self._elapsed_timer is None:
            return
        self._elapsed_timer.stop()
        self._elapsed_timer = None

    def _refresh_elapsed_tooltip(self) -> None:
        if self._elapsed_provider is None:
            return
        elapsed = self._elapsed_provider()
        self._tray.setToolTip(
            f"{_TITLE_PREFIX}Recording {format_duration(elapsed)}"
        )

    # Exposed for tests/diagnostics — not part of the TrayIcon Protocol.
    def current_icon(self) -> QIcon:
        return self._icon

    def current_title(self) -> str:
        return self._title

    def run(self, setup: Callable[["QtTrayAdapter"], None]) -> None:
        setup(self)
        _run_event_loop()

    def stop(self) -> None:
        self.visible = False
        app = QApplication.instance()
        if app is not None:
            app.quit()


def build_qt_tray_adapter(
    *,
    initial_state: AppState,
    get_state_label: Callable[[], str],
    on_settings_clicked: Callable[[], None] | None,
    on_quit_clicked: Callable[[], None],
) -> QtTrayAdapter:
    """Factory matching :data:`voicetotext.tray.app.TrayIconFactory`.

    Injected into :class:`SystemTrayApp` so the tray-lib details stay
    out of the composition root's import graph — bootstrap hands over
    this factory, the app calls it once inside :meth:`SystemTrayApp.run`.
    """
    return QtTrayAdapter(
        initial_state=initial_state,
        initial_label=get_state_label(),
        get_state_label=get_state_label,
        on_settings_clicked=on_settings_clicked,
        on_quit_clicked=on_quit_clicked,
    )
