import pytest

from voicetotext.core.events import StateChangedEvent
from voicetotext.settings.qt_state_bridge import QtStateBridge


@pytest.fixture
def bridge(qtbot):
    return QtStateBridge()


class TestQtStateBridgeSignals:
    def test_should_emit_state_changed_signal(self, bridge):
        received = []
        bridge.state_changed.connect(received.append)

        event = StateChangedEvent(old_state="idle", new_state="recording")
        bridge.on_state_changed(event)

        assert received == [event]
