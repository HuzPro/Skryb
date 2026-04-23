"""Regression tests for the ``open_settings`` lifecycle.

These guard the fix for the crash

    ``RuntimeError: Signal source has been deleted``

that happened when a Python-GC-destroyed settings window left its
bridges subscribed to :class:`SettingsService` and the next worker-
thread fanout tried to call into the dead QObject wrapper.

The tests work at the :class:`QtSettingsWindow` seam — closing a
window must leave the service's listener lists empty, and opening a
fresh window must reinstate exactly one listener on each seam. A
bootstrap-level end-to-end test would require the full composition
root, so the guard on the ``destroyed`` handler is covered by
``TestMakeDestroyedHandler`` in ``tests/test_bootstrap.py``.
"""
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import Qt

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.qt_window import QtSettingsWindow
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft


def _draft() -> SettingsDraft:
    return SettingsDraft(
        model_size="base",
        language="en",
        device="auto",
        hotkey_combination="ctrl+alt+m",
        hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE,
    )


class _TrackingService:
    """Stand-in for :class:`SettingsService` that records subscriptions."""

    def __init__(self) -> None:
        self._engine_listeners: list[object] = []
        self._state_listeners: list[object] = []
        self.list_models = MagicMock(return_value=[])
        self.selected_model = MagicMock(return_value="base")
        self.current_draft = MagicMock(return_value=_draft())
        self.any_model_installed = MagicMock(return_value=True)
        self.can_apply_now = MagicMock(return_value=True)
        self.is_engine_swapping = MagicMock(return_value=False)
        self.apply_draft = MagicMock(
            return_value=ChangeSet(
                engine_reload_required=False,
                hotkey_changed=False,
                output_changed=False,
            )
        )
        self.trigger_engine_swap = MagicMock()

    def subscribe_engine(self, listener: object) -> None:
        self._engine_listeners.append(listener)

    def unsubscribe_engine(self, listener: object) -> None:
        if listener in self._engine_listeners:
            self._engine_listeners.remove(listener)

    def subscribe_state_changes(self, listener: object) -> None:
        self._state_listeners.append(listener)

    def unsubscribe_state_changes(self, listener: object) -> None:
        if listener in self._state_listeners:
            self._state_listeners.remove(listener)

    def engine_listener_count(self) -> int:
        return len(self._engine_listeners)

    def state_listener_count(self) -> int:
        return len(self._state_listeners)


@pytest.fixture
def service() -> _TrackingService:
    return _TrackingService()


class TestOpenSettingsLifecycle:
    def test_should_subscribe_one_listener_per_seam_on_open(
        self, qtbot, service
    ):
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        assert service.engine_listener_count() == 1
        assert service.state_listener_count() == 1

    def test_should_drop_listeners_on_close(self, qtbot, service):
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        with qtbot.waitSignal(window.destroyed, timeout=1000):
            window.close()

        assert service.engine_listener_count() == 0
        assert service.state_listener_count() == 0

    def test_should_not_leak_listeners_across_open_close_reopen(
        self, qtbot, service
    ):
        window1 = QtSettingsWindow(service)
        qtbot.addWidget(window1)
        with qtbot.waitSignal(window1.destroyed, timeout=1000):
            window1.close()

        window2 = QtSettingsWindow(service)
        qtbot.addWidget(window2)

        assert service.engine_listener_count() == 1
        assert service.state_listener_count() == 1

    def test_should_keep_exactly_one_listener_after_save_triggered_close(
        self, qtbot, service
    ):
        service.apply_draft.return_value = ChangeSet(
            engine_reload_required=False,
            hotkey_changed=False,
            output_changed=False,
        )
        window = QtSettingsWindow(service)
        qtbot.addWidget(window)

        with qtbot.waitSignal(window.destroyed, timeout=1000):
            qtbot.mouseClick(window.save_button(), Qt.LeftButton)

        assert service.engine_listener_count() == 0
        assert service.state_listener_count() == 0
