import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
)
from voicetotext.downloads.state import DownloadState, DownloadStats


_SPEED_WINDOW_SECONDS = 3.0


class DownloadJob(Protocol):
    """What the manager runs: :class:`ModelDownloadJob` or any job shaped like it."""

    @property
    def job_id(self) -> str: ...

    @property
    def total_bytes(self) -> int: ...

    def run(
        self,
        on_progress: Callable[[int, int], None],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None: ...


class DownloadHandle:
    """Caller-facing command interface for one enqueued job."""

    def __init__(self, job_id: str, manager: "DownloadManager") -> None:
        self._job_id = job_id
        self._manager = manager

    @property
    def id(self) -> str:
        return self._job_id

    def state(self) -> DownloadState:
        return self._manager.state_of(self._job_id)

    def stats(self) -> DownloadStats:
        return self._manager.stats_of(self._job_id)

    def pause(self) -> None:
        self._manager.pause(self._job_id)

    def resume(self) -> None:
        self._manager.resume(self._job_id)

    def cancel(self) -> None:
        self._manager.cancel(self._job_id)


@dataclass
class _ActiveJob:
    job: DownloadJob
    state: DownloadState = DownloadState.QUEUED
    cancel_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    bytes_done: int = 0
    bytes_total: int = 0
    speed_samples: list[tuple[float, int]] = field(default_factory=list)
    future: Future[None] | None = None


def _current_speed(active: _ActiveJob) -> float:
    samples = active.speed_samples
    if len(samples) < 2:
        return 0.0
    t0, b0 = samples[0]
    t1, b1 = samples[-1]
    if t1 <= t0:
        return 0.0
    return (b1 - b0) / (t1 - t0)


def _current_eta(active: _ActiveJob) -> float | None:
    speed = _current_speed(active)
    if speed <= 0 or active.bytes_total <= 0:
        return None
    remaining = active.bytes_total - active.bytes_done
    if remaining <= 0:
        return 0.0
    return remaining / speed


def _is_in_flight(state: DownloadState) -> bool:
    """Whether a job is actively queued or running (not re-enqueuable).

    Any other state - PAUSED, FAILED, CANCELLED, COMPLETED - is terminal
    or idle, so a fresh ``enqueue`` of the same id must start a new run
    rather than no-op. Only an in-flight job is protected from a duplicate
    submit.
    """
    return state in (DownloadState.QUEUED, DownloadState.DOWNLOADING)


class DownloadManager:
    """Bounded thread pool that runs :class:`DownloadJob` instances and
    publishes their lifecycle on the shared :class:`EventBus`.

    Exposes pause/resume/cancel via :class:`DownloadHandle`.
    """

    def __init__(
        self,
        event_bus: EventBus,
        max_concurrent_downloads: int = 3,
    ) -> None:
        self._event_bus = event_bus
        self._pool = ThreadPoolExecutor(max_workers=max_concurrent_downloads)
        self._jobs: dict[str, _ActiveJob] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, job: DownloadJob) -> DownloadHandle:
        with self._lock:
            existing = self._jobs.get(job.job_id)
            if existing is not None and _is_in_flight(existing.state):
                return DownloadHandle(job.job_id, self)
            # Terminal/idle (or never-seen) job: start a fresh run. Replacing
            # the _ActiveJob - rather than mutating it - zeroes the progress
            # counters and hands out new cancel/pause Events, so a re-download
            # after cancel/complete/fail starts clean and a still-running
            # orphan worker can only mutate the object we just dropped.
            active = _ActiveJob(job=job, bytes_total=job.total_bytes)
            self._jobs[job.job_id] = active
            active.future = self._pool.submit(self._run_job, active)
        return DownloadHandle(job.job_id, self)

    def pause(self, job_id: str) -> None:
        with self._lock:
            active = self._jobs.get(job_id)
            if active is None or active.state != DownloadState.DOWNLOADING:
                return
            active.pause_event.set()

    def resume(self, job_id: str) -> None:
        with self._lock:
            active = self._jobs.get(job_id)
            if active is None or active.state != DownloadState.PAUSED:
                return
            active.pause_event.clear()
            active.future = self._pool.submit(self._run_job, active)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            active = self._jobs.get(job_id)
            if active is None:
                return
            active.cancel_event.set()

    def shutdown(self) -> None:
        with self._lock:
            for active in self._jobs.values():
                active.cancel_event.set()
        # ``wait=False`` so a stuck HTTP read in a worker thread cannot
        # block the caller (typically the tray-quit handler). The
        # process is about to ``os._exit`` anyway, which terminates any
        # surviving worker thread along with everything else.
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # Handle accessors
    # ------------------------------------------------------------------

    def state_of(self, job_id: str) -> DownloadState:
        with self._lock:
            active = self._jobs.get(job_id)
            return active.state if active else DownloadState.FAILED

    def stats_of(self, job_id: str) -> DownloadStats:
        with self._lock:
            active = self._jobs.get(job_id)
            if active is None:
                return DownloadStats(0, 0, 0.0, None)
            return DownloadStats(
                bytes_done=active.bytes_done,
                bytes_total=active.bytes_total,
                speed_bps=_current_speed(active),
                eta_seconds=_current_eta(active),
            )

    # ------------------------------------------------------------------
    # Worker entry point
    # ------------------------------------------------------------------

    def _run_job(self, active: _ActiveJob) -> None:
        self._transition(active, DownloadState.DOWNLOADING)
        try:
            active.job.run(
                on_progress=lambda done, total: self._on_progress(active, done, total),
                cancel_event=active.cancel_event,
                pause_event=active.pause_event,
            )
        except Exception as exc:
            if active.cancel_event.is_set():
                self._transition(active, DownloadState.CANCELLED)
            elif active.pause_event.is_set():
                self._transition(active, DownloadState.PAUSED)
            else:
                self._transition(active, DownloadState.FAILED)
                self._event_bus.publish(
                    DownloadFailedEvent(job_id=active.job.job_id, error=str(exc))
                )
            return

        # Job returned normally - check for silent pause/cancel signals.
        if active.cancel_event.is_set():
            self._transition(active, DownloadState.CANCELLED)
            return
        if active.pause_event.is_set():
            self._transition(active, DownloadState.PAUSED)
            return

        self._transition(active, DownloadState.COMPLETED)
        self._event_bus.publish(DownloadCompletedEvent(job_id=active.job.job_id))

    def _on_progress(self, active: _ActiveJob, done: int, total: int) -> None:
        with self._lock:
            active.bytes_done = done
            active.bytes_total = total
            now = time.monotonic()
            active.speed_samples.append((now, done))
            cutoff = now - _SPEED_WINDOW_SECONDS
            active.speed_samples = [
                s for s in active.speed_samples if s[0] >= cutoff
            ]
            speed = _current_speed(active)
            eta = _current_eta(active)

        self._event_bus.publish(
            DownloadProgressEvent(
                job_id=active.job.job_id,
                bytes_done=done,
                bytes_total=total,
                speed_bps=speed,
                eta_seconds=eta,
            )
        )

    def _transition(self, active: _ActiveJob, new_state: DownloadState) -> None:
        with self._lock:
            active.state = new_state
        self._event_bus.publish(
            DownloadStateChangedEvent(
                job_id=active.job.job_id,
                state=new_state.value,
            )
        )
