import threading
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock


from skryb.core.event_bus import SimpleEventBus
from skryb.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
)
from skryb.downloads.download_manager import DownloadManager
from skryb.downloads.hf_repo_resolver import ResolvedFile
from skryb.downloads.model_download_job import ModelDownloadJob
from skryb.downloads.state import DownloadState


@dataclass
class _SlowDownloader:
    """Stand-in for ChunkedFileDownloader that takes a while to finish so we
    can observe intermediate state."""

    payload: bytes = b"X" * 100
    delay_seconds: float = 0.05

    def download(self, plan, on_progress, cancel_event, pause_event):
        plan.target.parent.mkdir(parents=True, exist_ok=True)
        plan.target.write_bytes(self.payload)
        time.sleep(self.delay_seconds)
        on_progress(len(self.payload))


def _make_job(
    tmp_path: Path,
    job_id: str = "test-model",
    downloader=None,
    files=None,
) -> ModelDownloadJob:
    resolved = files or [ResolvedFile(filename="f.bin", size=100, url="u/f")]
    return ModelDownloadJob(
        job_id=job_id,
        repo_id="org/repo",
        commit_sha="sha",
        files=resolved,
        models_dir=tmp_path,
        file_downloader=downloader or _SlowDownloader(),
    )


def _wait_for(predicate, timeout: float = 2.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("Predicate did not become true within timeout")


class TestDownloadManagerEnqueue:
    def test_enqueue_should_return_handle_with_queued_state(self, tmp_path):
        manager = DownloadManager(
            event_bus=SimpleEventBus(), max_concurrent_downloads=1
        )
        job = _make_job(tmp_path)

        handle = manager.enqueue(job)

        try:
            assert handle.id == "test-model"
            assert handle.state() in (DownloadState.QUEUED, DownloadState.DOWNLOADING, DownloadState.COMPLETED)
        finally:
            manager.shutdown()

    def test_enqueue_should_run_job_and_publish_completed_event(self, tmp_path):
        bus = SimpleEventBus()
        completed: list = []
        bus.subscribe(DownloadCompletedEvent, completed.append)

        manager = DownloadManager(event_bus=bus, max_concurrent_downloads=1)
        job = _make_job(tmp_path)

        handle = manager.enqueue(job)
        _wait_for(lambda: handle.state() == DownloadState.COMPLETED)

        assert len(completed) == 1
        assert completed[0].job_id == "test-model"
        manager.shutdown()

    def test_enqueue_should_publish_progress_events(self, tmp_path):
        bus = SimpleEventBus()
        progress: list = []
        bus.subscribe(DownloadProgressEvent, progress.append)

        manager = DownloadManager(event_bus=bus, max_concurrent_downloads=1)
        job = _make_job(tmp_path)

        handle = manager.enqueue(job)
        _wait_for(lambda: handle.state() == DownloadState.COMPLETED)

        assert len(progress) >= 1
        assert progress[-1].bytes_done == 100
        assert progress[-1].bytes_total == 100
        manager.shutdown()

    def test_enqueue_should_publish_state_changes(self, tmp_path):
        bus = SimpleEventBus()
        states: list = []
        bus.subscribe(DownloadStateChangedEvent, lambda e: states.append(e.state))

        manager = DownloadManager(event_bus=bus, max_concurrent_downloads=1)
        job = _make_job(tmp_path)
        handle = manager.enqueue(job)

        _wait_for(lambda: handle.state() == DownloadState.COMPLETED)

        assert "downloading" in states
        assert "completed" in states
        manager.shutdown()


class TestDownloadManagerFailure:
    def test_should_publish_failed_event_when_job_raises(self, tmp_path):
        bus = SimpleEventBus()
        failed: list = []
        bus.subscribe(DownloadFailedEvent, failed.append)

        broken = MagicMock()
        broken.download.side_effect = RuntimeError("nope")
        manager = DownloadManager(event_bus=bus, max_concurrent_downloads=1)
        job = _make_job(tmp_path, downloader=broken)

        handle = manager.enqueue(job)
        _wait_for(lambda: handle.state() == DownloadState.FAILED)

        assert len(failed) == 1
        assert "nope" in failed[0].error
        manager.shutdown()


class TestDownloadManagerPauseResume:
    def test_pause_should_set_paused_state(self, tmp_path):
        bus = SimpleEventBus()

        # Job that takes long enough to be paused mid-run
        class _HangingDownloader:
            def __init__(self):
                self.started = threading.Event()

            def download(self, plan, on_progress, cancel_event, pause_event):
                self.started.set()
                while not pause_event.is_set() and not cancel_event.is_set():
                    time.sleep(0.01)

        downloader = _HangingDownloader()
        manager = DownloadManager(event_bus=bus, max_concurrent_downloads=1)
        job = _make_job(tmp_path, downloader=downloader)
        handle = manager.enqueue(job)

        downloader.started.wait(timeout=2)
        handle.pause()

        _wait_for(lambda: handle.state() == DownloadState.PAUSED)
        manager.shutdown()

    def test_cancel_should_set_cancelled_state(self, tmp_path):
        class _HangingDownloader:
            def __init__(self):
                self.started = threading.Event()

            def download(self, plan, on_progress, cancel_event, pause_event):
                self.started.set()
                while not cancel_event.is_set():
                    time.sleep(0.01)
                raise RuntimeError("cancelled")

        downloader = _HangingDownloader()
        manager = DownloadManager(
            event_bus=SimpleEventBus(), max_concurrent_downloads=1
        )
        job = _make_job(tmp_path, downloader=downloader)
        handle = manager.enqueue(job)

        downloader.started.wait(timeout=2)
        handle.cancel()

        _wait_for(lambda: handle.state() == DownloadState.CANCELLED)
        manager.shutdown()


class _HangingDownloader:
    """Blocks until cancelled/paused so a job can be observed mid-flight."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.calls = 0

    def download(self, plan, on_progress, cancel_event, pause_event):
        self.calls += 1
        self.started.set()
        while not pause_event.is_set() and not cancel_event.is_set():
            time.sleep(0.005)
        raise RuntimeError("interrupted")


class TestDownloadManagerRestart:
    """A re-download with the same job_id must start a fresh run once the
    previous one reached a terminal state (CANCELLED, COMPLETED, FAILED),
    not silently no-op. Only an in-flight job (QUEUED/DOWNLOADING) is
    protected from a duplicate submit.
    """

    def test_enqueue_should_rerun_a_cancelled_job_when_re_enqueued_with_same_id(
        self, tmp_path,
    ):
        bus = SimpleEventBus()
        completed: list = []
        bus.subscribe(DownloadCompletedEvent, completed.append)
        manager = DownloadManager(event_bus=bus, max_concurrent_downloads=1)

        hanging = _HangingDownloader()
        handle = manager.enqueue(_make_job(tmp_path, downloader=hanging))
        hanging.started.wait(timeout=2)
        handle.cancel()
        _wait_for(lambda: handle.state() == DownloadState.CANCELLED)

        # Re-download the same model with a downloader that finishes.
        manager.enqueue(_make_job(tmp_path, downloader=_SlowDownloader()))
        _wait_for(lambda: handle.state() == DownloadState.COMPLETED)

        assert any(e.job_id == "test-model" for e in completed)
        manager.shutdown()

    def test_enqueue_should_rerun_a_completed_job_when_re_enqueued_with_same_id(
        self, tmp_path,
    ):
        bus = SimpleEventBus()
        completed: list = []
        bus.subscribe(DownloadCompletedEvent, completed.append)
        manager = DownloadManager(event_bus=bus, max_concurrent_downloads=1)

        handle = manager.enqueue(_make_job(tmp_path))
        _wait_for(lambda: handle.state() == DownloadState.COMPLETED)
        assert len(completed) == 1

        # Re-download (e.g. after the model was deleted from disk).
        manager.enqueue(_make_job(tmp_path))
        _wait_for(lambda: len(completed) == 2)

        manager.shutdown()

    def test_enqueue_should_be_a_noop_while_job_is_still_downloading(
        self, tmp_path,
    ):
        manager = DownloadManager(
            event_bus=SimpleEventBus(), max_concurrent_downloads=2,
        )
        hanging = _HangingDownloader()
        manager.enqueue(_make_job(tmp_path, downloader=hanging))
        hanging.started.wait(timeout=2)

        # Same id while in-flight: must not start a second worker.
        manager.enqueue(_make_job(tmp_path, downloader=hanging))
        time.sleep(0.1)

        assert hanging.calls == 1
        manager.shutdown()

    def test_re_enqueue_after_cancel_should_reset_progress_counters_to_zero(
        self, tmp_path,
    ):
        manager = DownloadManager(
            event_bus=SimpleEventBus(), max_concurrent_downloads=1,
        )

        class _ProgressThenHang:
            def __init__(self) -> None:
                self.started = threading.Event()

            def download(self, plan, on_progress, cancel_event, pause_event):
                on_progress(50)
                self.started.set()
                while not cancel_event.is_set():
                    time.sleep(0.005)
                raise RuntimeError("cancelled")

        first = _ProgressThenHang()
        handle = manager.enqueue(_make_job(tmp_path, downloader=first))
        first.started.wait(timeout=2)
        _wait_for(lambda: handle.stats().bytes_done == 50)
        handle.cancel()
        _wait_for(lambda: handle.state() == DownloadState.CANCELLED)

        # A fresh run replaces the _ActiveJob, so counters start at zero
        # rather than showing the stale partial fill from the cancelled run.
        blocker = _HangingDownloader()
        manager.enqueue(_make_job(tmp_path, downloader=blocker))
        blocker.started.wait(timeout=2)

        assert handle.stats().bytes_done == 0
        assert handle.stats().speed_bps == 0.0
        manager.shutdown()


class TestDownloadManagerConcurrency:
    def test_should_run_up_to_max_concurrent_jobs_simultaneously(self, tmp_path):
        in_flight = 0
        peak = 0
        lock = threading.Lock()

        class _TrackingDownloader:
            def download(self, plan, on_progress, cancel_event, pause_event):
                nonlocal in_flight, peak
                with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                time.sleep(0.05)
                with lock:
                    in_flight -= 1
                plan.target.parent.mkdir(parents=True, exist_ok=True)
                plan.target.write_bytes(b"x" * 100)
                on_progress(100)

        manager = DownloadManager(
            event_bus=SimpleEventBus(), max_concurrent_downloads=3
        )
        downloader = _TrackingDownloader()
        handles = [
            manager.enqueue(_make_job(tmp_path, job_id=f"job{i}", downloader=downloader))
            for i in range(5)
        ]

        for h in handles:
            _wait_for(lambda h=h: h.state() == DownloadState.COMPLETED, timeout=5)

        assert peak <= 3
        assert peak >= 2  # at least 2 ran in parallel
        manager.shutdown()


class _GatedJob:
    """A job that holds its slot until the test opens the gate."""

    def __init__(self, job_id: str, gate: threading.Event, started: list[str]) -> None:
        self.job_id = job_id
        self.total_bytes = 100
        self._gate = gate
        self._started = started

    def run(self, on_progress, cancel_event, pause_event) -> None:
        self._started.append(self.job_id)
        self._gate.wait(5)
        on_progress(100, 100)


class TestDownloadsPastTheLimitWaitTheirTurn:
    def test_five_downloads_run_at_once_by_default(self):
        gate, started = threading.Event(), []
        manager = DownloadManager(event_bus=SimpleEventBus())
        try:
            for index in range(7):
                manager.enqueue(_GatedJob(f"model-{index}", gate, started))
            _wait_for(lambda: len(started) == 5)
            time.sleep(0.05)
            assert len(started) == 5
        finally:
            gate.set()
            manager.shutdown()

    def test_every_click_is_announced_as_queued_at_once(self):
        bus = SimpleEventBus()
        states: list[tuple[str, str]] = []
        bus.subscribe(DownloadStateChangedEvent, lambda event: states.append((event.job_id, event.state)))
        gate, started = threading.Event(), []
        manager = DownloadManager(event_bus=bus, max_concurrent_downloads=5)
        names = ["parakeet-tdt-0.6b-v3", "parakeet-ultra", "nemotron-speech-streaming-en-0.6b", "qwen3-asr-1.7b",
                 "crisperwhisper-2.0-large", "phonon-2", "crisperwhisper-2.0-turbo"]
        try:
            for name in names:
                manager.enqueue(_GatedJob(name, gate, started))
            assert [job for job, state in states if state == "queued"] == names
            _wait_for(lambda: len(started) == 5)
            assert manager.state_of("phonon-2") is DownloadState.QUEUED
            assert manager.state_of("crisperwhisper-2.0-turbo") is DownloadState.QUEUED
        finally:
            gate.set()
            manager.shutdown()

    def test_a_queued_download_starts_when_a_slot_frees(self):
        first_gate, second_gate, started = threading.Event(), threading.Event(), []
        manager = DownloadManager(event_bus=SimpleEventBus(), max_concurrent_downloads=1)
        try:
            manager.enqueue(_GatedJob("first", first_gate, started))
            manager.enqueue(_GatedJob("second", second_gate, started))
            _wait_for(lambda: started == ["first"])

            first_gate.set()

            _wait_for(lambda: started == ["first", "second"])
        finally:
            second_gate.set()
            manager.shutdown()

    def test_cancelling_a_queued_download_never_starts_it(self):
        bus = SimpleEventBus()
        states: list[str] = []
        bus.subscribe(DownloadStateChangedEvent, lambda event: event.job_id == "waiting" and states.append(event.state))
        gate, started = threading.Event(), []
        manager = DownloadManager(event_bus=bus, max_concurrent_downloads=1)
        try:
            manager.enqueue(_GatedJob("running", gate, started))
            manager.enqueue(_GatedJob("waiting", threading.Event(), started))
            manager.cancel("waiting")

            gate.set()

            _wait_for(lambda: manager.state_of("waiting") is DownloadState.CANCELLED)
            assert started == ["running"]
            assert states == ["queued", "cancelled"]
        finally:
            manager.shutdown()
