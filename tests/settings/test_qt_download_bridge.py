import pytest

from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    ModelDeletedEvent,
)
from voicetotext.settings.qt_download_bridge import QtDownloadBridge


@pytest.fixture
def bridge(qtbot):
    widget = QtDownloadBridge()
    return widget


class TestQtDownloadBridgeSignals:
    def test_should_emit_progress_signal_when_listener_receives_progress(self, bridge):
        received = []
        bridge.progress.connect(received.append)

        event = DownloadProgressEvent(
            job_id="base",
            bytes_done=100,
            bytes_total=1000,
            speed_bps=50.0,
            eta_seconds=18.0,
        )
        bridge.on_download_progress(event)

        assert received == [event]

    def test_should_emit_state_signal_when_listener_receives_state_change(self, bridge):
        received = []
        bridge.state.connect(received.append)

        event = DownloadStateChangedEvent(job_id="base", state="paused")
        bridge.on_download_state(event)

        assert received == [event]

    def test_should_emit_completed_signal_when_listener_receives_completed(self, bridge):
        received = []
        bridge.completed.connect(received.append)

        event = DownloadCompletedEvent(job_id="base")
        bridge.on_download_completed(event)

        assert received == [event]

    def test_should_emit_failed_signal_when_listener_receives_failure(self, bridge):
        received = []
        bridge.failed.connect(received.append)

        event = DownloadFailedEvent(job_id="base", error="disk full")
        bridge.on_download_failed(event)

        assert received == [event]

    def test_should_re_emit_model_deleted_event_as_qt_signal(self, qtbot, bridge):
        with qtbot.waitSignal(bridge.deleted, timeout=500) as sig:
            bridge.on_model_deleted(ModelDeletedEvent(name="tiny"))

        assert sig.args[0].name == "tiny"
