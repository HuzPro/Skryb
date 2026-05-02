from unittest.mock import MagicMock

from voicetotext.core.event_bus import SimpleEventBus
from voicetotext.core.events import (
    ErrorOccurredEvent,
    StateChangedEvent,
    TranscriptionFailedEvent,
)
from voicetotext.core.state_machine import AppState, AppStateMachine
from voicetotext.tray.app import SystemTrayApp


def _build_app(on_open_settings=None, tray_factory=None):
    use_case = MagicMock()
    event_bus = SimpleEventBus()
    state_machine = AppStateMachine(event_bus=event_bus)
    listener = MagicMock()
    return SystemTrayApp(
        use_case=use_case,
        state_machine=state_machine,
        event_bus=event_bus,
        hotkey_listener=listener,
        tray_factory=tray_factory or (lambda **kwargs: MagicMock()),
        on_open_settings=on_open_settings,
    )


class TestSettingsClick:
    def test_should_call_open_settings_when_click_handled(self):
        call_count = 0

        def on_open() -> None:
            nonlocal call_count
            call_count += 1

        app = _build_app(on_open_settings=on_open)

        app._handle_settings_clicked()

        assert call_count == 1

    def test_should_noop_when_no_settings_callback_registered(self):
        app = _build_app(on_open_settings=None)

        # No callback wired: click must not raise.
        app._handle_settings_clicked()


class TestTrayIconFactoryInvocation:
    def test_should_invoke_factory_with_callbacks_on_run(self):
        captured: dict = {}

        def factory(**kwargs):
            captured.update(kwargs)
            icon = MagicMock()
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(on_open_settings=lambda: None, tray_factory=factory)

        app.run()

        assert callable(captured["get_state_label"])
        assert callable(captured["on_settings_clicked"])
        assert callable(captured["on_quit_clicked"])
        assert captured["initial_state"] == AppState.IDLE


class TestStateChangeForwarding:
    def test_should_call_update_state_on_icon_when_state_changes(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._handle_state_changed(
            StateChangedEvent(
                old_state=AppState.IDLE.value,
                new_state=AppState.RECORDING.value,
            )
        )

        icon.update_state.assert_called_with(AppState.RECORDING, "Recording...")


class TestErrorNotification:
    def test_should_notify_on_error_occurred_event(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._event_bus.publish(
            ErrorOccurredEvent(error="Clipboard write failed.")
        )

        icon.notify.assert_called_once_with(
            "VoiceToText", "Clipboard write failed."
        )

    def test_should_notify_on_transcription_failed_event(self):
        icon = MagicMock()

        def factory(**kwargs):
            icon.run.side_effect = lambda setup: setup(icon)
            return icon

        app = _build_app(tray_factory=factory)
        app.run()

        app._event_bus.publish(
            TranscriptionFailedEvent(error="model crashed")
        )

        icon.notify.assert_called_once_with("VoiceToText", "model crashed")
