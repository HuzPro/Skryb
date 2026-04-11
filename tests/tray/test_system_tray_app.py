import threading
import time
from unittest.mock import MagicMock

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.state_machine import AppStateMachine
from voicetotext.tray.app import SystemTrayApp


def _build_app(on_open_settings=None):
    use_case = MagicMock()
    event_bus = SimpleEventBus()
    state_machine = AppStateMachine(event_bus=event_bus)
    listener = MagicMock()
    return SystemTrayApp(
        use_case=use_case,
        state_machine=state_machine,
        event_bus=event_bus,
        hotkey_listener=listener,
        on_open_settings=on_open_settings,
    )


class TestSettingsWindowSingleInstance:
    def test_should_ignore_second_click_while_settings_is_open(self):
        call_count = 0
        barrier = threading.Event()

        def slow_open():
            nonlocal call_count
            call_count += 1
            barrier.wait(timeout=5)

        app = _build_app(on_open_settings=slow_open)
        icon = MagicMock()

        app._on_settings_clicked(icon, None)
        time.sleep(0.05)
        app._on_settings_clicked(icon, None)
        time.sleep(0.05)

        assert call_count == 1

        barrier.set()

    def test_should_allow_reopening_after_settings_closes(self):
        call_count = 0

        def fast_open():
            nonlocal call_count
            call_count += 1

        app = _build_app(on_open_settings=fast_open)
        icon = MagicMock()

        app._on_settings_clicked(icon, None)
        time.sleep(0.05)
        app._on_settings_clicked(icon, None)
        time.sleep(0.05)

        assert call_count == 2
