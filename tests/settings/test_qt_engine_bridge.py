import pytest

from voicetotext.core.events import (
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
)
from voicetotext.settings.qt_engine_bridge import QtEngineBridge


@pytest.fixture
def bridge(qtbot):
    return QtEngineBridge()


class TestQtEngineBridgeSignals:
    def test_should_emit_load_started_signal(self, bridge):
        received = []
        bridge.load_started.connect(received.append)

        event = EngineLoadStartedEvent(model_name="small")
        bridge.on_engine_load_started(event)

        assert received == [event]

    def test_should_emit_load_completed_signal(self, bridge):
        received = []
        bridge.load_completed.connect(received.append)

        event = EngineLoadCompletedEvent(model_name="small")
        bridge.on_engine_load_completed(event)

        assert received == [event]

    def test_should_emit_load_failed_signal(self, bridge):
        received = []
        bridge.load_failed.connect(received.append)

        event = EngineLoadFailedEvent(model_name="small", error="oom")
        bridge.on_engine_load_failed(event)

        assert received == [event]
