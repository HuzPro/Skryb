from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.settings.service import EngineEventListener
from voicetotext.settings.tk_engine_bridge import TkEngineBridge


class _FakeWindow:
    def __init__(self) -> None:
        self.started: list[EngineLoadStartedEvent] = []
        self.completed: list[EngineLoadCompletedEvent] = []
        self.failed: list[EngineLoadFailedEvent] = []

    def on_engine_load_started(self, event: EngineLoadStartedEvent) -> None:
        self.started.append(event)

    def on_engine_load_completed(self, event: EngineLoadCompletedEvent) -> None:
        self.completed.append(event)

    def on_engine_load_failed(self, event: EngineLoadFailedEvent) -> None:
        self.failed.append(event)


class TestTkEngineBridge:
    def test_should_be_an_engine_event_listener(self) -> None:
        bridge = TkEngineBridge(schedule=lambda action: None, target=_FakeWindow())

        assert isinstance(bridge, EngineEventListener)

    def test_should_forward_load_started_when_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkEngineBridge(schedule=scheduled.append, target=window)
        event = EngineLoadStartedEvent(model_name="small")

        bridge.on_engine_load_started(event)
        assert window.started == []
        scheduled[0]()

        assert window.started == [event]

    def test_should_forward_load_completed_when_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkEngineBridge(schedule=scheduled.append, target=window)
        event = EngineLoadCompletedEvent(model_name="small")

        bridge.on_engine_load_completed(event)
        scheduled[0]()

        assert window.completed == [event]

    def test_should_forward_load_failed_when_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkEngineBridge(schedule=scheduled.append, target=window)
        event = EngineLoadFailedEvent(model_name="small", error="oom")

        bridge.on_engine_load_failed(event)
        scheduled[0]()

        assert window.failed == [event]
