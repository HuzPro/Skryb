from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    ModelDeletedEvent,
)
from voicetotext.settings.service import DownloadEventListener
from voicetotext.settings.tk_download_bridge import TkDownloadBridge


class _FakeWindow:
    def __init__(self) -> None:
        self.progress: list[DownloadProgressEvent] = []
        self.state: list[DownloadStateChangedEvent] = []
        self.completed: list[DownloadCompletedEvent] = []
        self.failed: list[DownloadFailedEvent] = []
        self.deleted: list[ModelDeletedEvent] = []

    def on_download_progress(self, event: DownloadProgressEvent) -> None:
        self.progress.append(event)

    def on_download_state(self, event: DownloadStateChangedEvent) -> None:
        self.state.append(event)

    def on_download_completed(self, event: DownloadCompletedEvent) -> None:
        self.completed.append(event)

    def on_download_failed(self, event: DownloadFailedEvent) -> None:
        self.failed.append(event)

    def on_model_deleted(self, event: ModelDeletedEvent) -> None:
        self.deleted.append(event)


class TestTkDownloadBridge:
    def test_should_be_a_download_event_listener(self) -> None:
        bridge = TkDownloadBridge(schedule=lambda action: None, target=_FakeWindow())

        assert isinstance(bridge, DownloadEventListener)

    def test_should_forward_progress_when_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkDownloadBridge(schedule=scheduled.append, target=window)
        event = DownloadProgressEvent(
            job_id="parakeet-tdt-0.6b-v3",
            bytes_done=100,
            bytes_total=1000,
            speed_bps=50.0,
            eta_seconds=18.0,
        )

        bridge.on_download_progress(event)
        assert window.progress == []
        scheduled[0]()

        assert window.progress == [event]

    def test_should_forward_state_when_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkDownloadBridge(schedule=scheduled.append, target=window)
        event = DownloadStateChangedEvent(job_id="parakeet-tdt-0.6b-v3", state="paused")

        bridge.on_download_state(event)
        scheduled[0]()

        assert window.state == [event]

    def test_should_forward_completed_when_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkDownloadBridge(schedule=scheduled.append, target=window)
        event = DownloadCompletedEvent(job_id="parakeet-tdt-0.6b-v3")

        bridge.on_download_completed(event)
        scheduled[0]()

        assert window.completed == [event]

    def test_should_forward_failed_when_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkDownloadBridge(schedule=scheduled.append, target=window)
        event = DownloadFailedEvent(job_id="parakeet-tdt-0.6b-v3", error="disk full")

        bridge.on_download_failed(event)
        scheduled[0]()

        assert window.failed == [event]

    def test_should_forward_model_deleted_when_action_runs(self) -> None:
        scheduled: list = []
        window = _FakeWindow()
        bridge = TkDownloadBridge(schedule=scheduled.append, target=window)
        event = ModelDeletedEvent(name="test-model")

        bridge.on_model_deleted(event)
        scheduled[0]()

        assert window.deleted == [event]
