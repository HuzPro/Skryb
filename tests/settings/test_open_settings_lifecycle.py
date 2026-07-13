"""Regression tests for the ``open_settings`` lifecycle.

These guard against a settings window leaving its bridges subscribed to
:class:`SettingsService` after it closes, which would let the next
worker-thread fanout call into a torn-down window. The tests work at the
:class:`TkSettingsWindow` seam: closing a window must leave the service's
listener lists empty, and opening a fresh window must reinstate exactly
one listener on each seam.
"""
from unittest.mock import MagicMock

import pytest

from voicetotext.output.output_mode import OutputMode
from voicetotext.settings.settings_draft import ChangeSet, SettingsDraft
from voicetotext.settings.signal import Signal
from voicetotext.settings.tk_window import TkSettingsWindow


class FakeCaptureBridge:
    def __init__(self) -> None:
        self.key_pressed = Signal()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _draft() -> SettingsDraft:
    return SettingsDraft(
        model_size="base",
        language="en",
        device="auto",
        engine="auto",
        hotkey_combination="ctrl+alt+m",
        hotkey_mode="toggle",
        output_mode=OutputMode.DIRECT_PASTE,
    )


class _TrackingService:
    """Stand-in for :class:`SettingsService` that records subscriptions."""

    def __init__(self) -> None:
        self._engine_listeners: list[object] = []
        self._state_listeners: list[object] = []
        self._download_listeners: list[object] = []
        self._model_deleted_handlers: list[object] = []
        self.list_models = MagicMock(return_value=[])
        self.selected_model = MagicMock(return_value="base")
        self.current_draft = MagicMock(return_value=_draft())
        self.any_model_installed = MagicMock(return_value=True)
        self.is_installed = MagicMock(return_value=True)
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
        self.available_devices = MagicMock(return_value=["auto", "cuda", "cpu"])
        self.allows_device_selection = MagicMock(
            side_effect=lambda n: n in {"tiny", "base"}
        )
        self.supports_streaming_hardware = MagicMock(return_value=True)
        self.engine_candidates_for = MagicMock(return_value=())

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

    def subscribe_downloads(self, listener: object) -> None:
        self._download_listeners.append(listener)

    def unsubscribe_downloads(self, listener: object) -> None:
        if listener in self._download_listeners:
            self._download_listeners.remove(listener)

    def subscribe_model_deleted(self, handler: object) -> None:
        self._model_deleted_handlers.append(handler)

    def unsubscribe_model_deleted(self, handler: object) -> None:
        if handler in self._model_deleted_handlers:
            self._model_deleted_handlers.remove(handler)

    def engine_listener_count(self) -> int:
        return len(self._engine_listeners)

    def state_listener_count(self) -> int:
        return len(self._state_listeners)

    def download_listener_count(self) -> int:
        return len(self._download_listeners)

    def model_deleted_handler_count(self) -> int:
        return len(self._model_deleted_handlers)


@pytest.fixture
def service() -> _TrackingService:
    return _TrackingService()


def _make_window(tk_root, service) -> TkSettingsWindow:
    return TkSettingsWindow(
        tk_root, service, capture_bridge=FakeCaptureBridge(),
    )


def _destroy(window) -> None:
    if window.winfo_exists():
        window.destroy()


class TestOpenSettingsLifecycle:
    def test_should_subscribe_one_listener_per_seam_on_open(self, tk_root, service):
        window = _make_window(tk_root, service)

        assert service.engine_listener_count() == 1
        assert service.state_listener_count() == 1
        assert service.download_listener_count() == 1
        assert service.model_deleted_handler_count() == 1

        _destroy(window)

    def test_should_drop_listeners_on_close(self, tk_root, service):
        window = _make_window(tk_root, service)

        window.close()

        assert service.engine_listener_count() == 0
        assert service.state_listener_count() == 0
        assert service.download_listener_count() == 0
        assert service.model_deleted_handler_count() == 0

        _destroy(window)

    def test_should_not_leak_listeners_across_open_close_reopen(
        self, tk_root, service,
    ):
        window1 = _make_window(tk_root, service)
        window1.close()
        _destroy(window1)

        window2 = _make_window(tk_root, service)

        assert service.engine_listener_count() == 1
        assert service.state_listener_count() == 1
        assert service.download_listener_count() == 1
        assert service.model_deleted_handler_count() == 1

        _destroy(window2)

    def test_should_keep_no_listeners_after_save_triggered_close(
        self, tk_root, service,
    ):
        window = _make_window(tk_root, service)

        window.save_button().invoke()

        assert service.engine_listener_count() == 0
        assert service.state_listener_count() == 0
        assert service.download_listener_count() == 0
        assert service.model_deleted_handler_count() == 0

        _destroy(window)
