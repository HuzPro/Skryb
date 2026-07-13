from voicetotext.core.events import StateChangedEvent
from voicetotext.settings.service import StateChangeListener
from voicetotext.settings.tk_state_bridge import TkStateBridge


class _FakeWindow:
    def __init__(self) -> None:
        self.received: list[StateChangedEvent] = []

    def on_state_changed(self, event: StateChangedEvent) -> None:
        self.received.append(event)


class TestTkStateBridge:
    def test_should_be_a_state_change_listener(self) -> None:
        bridge = TkStateBridge(schedule=lambda action: None, target=_FakeWindow())

        assert isinstance(bridge, StateChangeListener)

    def test_should_defer_slot_until_scheduled_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkStateBridge(schedule=scheduled.append, target=window)

        bridge.on_state_changed(
            StateChangedEvent(old_state="idle", new_state="recording")
        )

        assert window.received == []
        assert len(scheduled) == 1

    def test_should_forward_event_to_target_when_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkStateBridge(schedule=scheduled.append, target=window)
        event = StateChangedEvent(old_state="idle", new_state="recording")

        bridge.on_state_changed(event)
        scheduled[0]()

        assert window.received == [event]
