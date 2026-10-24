"""Server-sent events for the UI shell.

The hub turns the app's own events into the contract's SSE events and hands
each one to every connected client, in the order it was published. Bus
handlers run on whichever thread published, so they only fill queues.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Protocol

from skryb.core.event_bus import EventBus
from skryb.core.events import (
    DomainEvent,
    DownloadCompletedEvent,
    DownloadFailedEvent,
    DownloadProgressEvent,
    DownloadStateChangedEvent,
    EngineLoadCompletedEvent,
    EngineLoadFailedEvent,
    EngineLoadStartedEvent,
    MicrophoneConnectedEvent,
    ModelDeletedEvent,
    RecordingCancelledEvent,
    StateChangedEvent,
    TranscriptionCompletedEvent,
    TranscriptionFailedEvent,
    WritingServerFailedEvent,
)
from skryb.core.state_machine import AppState

_logger = logging.getLogger(__name__)

Json = dict[str, object]
# ``None`` tells a client's stream to end.
SseEvent = tuple[str, object] | None

HEARTBEAT_SECONDS = 15.0
# Progress arrives once per network chunk; ten updates a second is plenty
# for a progress bar and keeps a fast download from flooding the shell.
_PROGRESS_SECONDS = 0.1
_HISTORY_EVENTS: tuple[type[DomainEvent], ...] = (
    TranscriptionCompletedEvent, TranscriptionFailedEvent, RecordingCancelledEvent,
)


class Snapshots(Protocol):
    """What the hub asks the API for when an event needs the whole picture."""

    def app_info(self) -> Json: ...
    def writing_status(self) -> Json | None: ...
    def watch_writing(self, listener: Callable[[], None]) -> Callable[[], None]: ...
    def release_devices(self) -> None: ...


class EventHub:
    def __init__(
        self,
        event_bus: EventBus,
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.heartbeat_seconds = heartbeat_seconds
        self._clock = clock
        # Re-entrant: greeting a new client asks the API for a snapshot, which
        # reads the engine status held under this same lock.
        self._lock = threading.RLock()
        self._clients: list[queue.Queue[SseEvent]] = []
        self._api: Snapshots | None = None
        self._stop_watching_writing: Callable[[], None] = _nothing
        self._engine: Json = {"loading": False, "ready": False, "error": None, "model": ""}
        self._downloads: dict[str, Json] = {}
        self._last_progress: dict[str, tuple[str, float]] = {}
        handlers: dict[type[DomainEvent], Callable[[DomainEvent], None]] = {
            StateChangedEvent: self._on_state_changed,
            EngineLoadStartedEvent: self._on_engine_event,
            EngineLoadCompletedEvent: self._on_engine_event,
            EngineLoadFailedEvent: self._on_engine_event,
            DownloadProgressEvent: self._on_download_event,
            DownloadStateChangedEvent: self._on_download_event,
            DownloadCompletedEvent: self._on_download_event,
            DownloadFailedEvent: self._on_download_event,
            ModelDeletedEvent: self._on_model_deleted,
            MicrophoneConnectedEvent: self._on_microphone_connected,
            WritingServerFailedEvent: lambda _event: self._publish_writing(),
            **{kind: self._on_history_changed for kind in _HISTORY_EVENTS},
        }
        for kind, handler in handlers.items():
            event_bus.subscribe(kind, handler)

    # -- wiring ----------------------------------------------------------

    def use(self, api: Snapshots, *, model: str, loading: bool) -> None:
        """Answer from *api* from now on; first-run setup hands over to the app this way."""
        self._stop_watching_writing()
        with self._lock:
            self._api = api
            self._engine = {"loading": loading, "ready": False, "error": None, "model": model}
        self._stop_watching_writing = api.watch_writing(self._publish_writing)
        self.publish("app", api.app_info())

    # -- clients -----------------------------------------------------------

    def connect(self) -> "queue.Queue[SseEvent]":
        client: queue.Queue[SseEvent] = queue.Queue()
        with self._lock:
            if self._api is not None:
                client.put(("app", self._api.app_info()))
            self._clients.append(client)
        return client

    def disconnect(self, client: "queue.Queue[SseEvent]") -> None:
        """Forget *client*; once none is left, let go of the microphone and keyboard.

        A shell that crashed mid-way would otherwise leave the level meter
        holding the microphone, or the shortcut recorder swallowing every key.
        """
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)
            api = None if self._clients else self._api
        if api is not None:
            _release_logging_failures(api)

    def close(self) -> None:
        with self._lock:
            clients, self._clients = self._clients, []
        for client in clients:
            client.put(None)

    def publish(self, name: str, data: object) -> None:
        with self._lock:
            for client in self._clients:
                client.put((name, data))

    # -- state the API reads -------------------------------------------------

    def engine_status(self) -> Json:
        with self._lock:
            return dict(self._engine)

    def download_of(self, name: str) -> Json | None:
        with self._lock:
            download = self._downloads.get(name)
            return None if download is None else {
                key: download[key] for key in ("state", "bytes_done", "bytes_total", "error")
            }

    # -- bus handlers ----------------------------------------------------------

    def _on_state_changed(self, event: DomainEvent) -> None:
        assert isinstance(event, StateChangedEvent)
        if event.old_state == AppState.LOADING.value and event.new_state == AppState.IDLE.value:
            self._set_engine(loading=False, ready=True, error=None)
        self._publish_app()

    def _on_engine_event(self, event: DomainEvent) -> None:
        if isinstance(event, EngineLoadStartedEvent):
            self._set_engine(loading=True, ready=False, error=None, model=event.model_name)
        elif isinstance(event, EngineLoadCompletedEvent):
            self._set_engine(loading=False, ready=True, error=None, model=event.model_name)
        elif isinstance(event, EngineLoadFailedEvent):
            self._set_engine(loading=False, ready=False, error=event.error, model=event.model_name)
        self._publish_app()

    def _set_engine(self, **changes: object) -> None:
        with self._lock:
            self._engine.update(changes)
            engine = dict(self._engine)
        state = "loading" if engine["loading"] else "ready" if engine["ready"] else "failed"
        self.publish("engine", {"state": state, **engine})

    def _on_download_event(self, event: DomainEvent) -> None:
        if isinstance(event, DownloadProgressEvent):
            self._record_download(
                event.job_id, "downloading",
                bytes_done=event.bytes_done, bytes_total=event.bytes_total,
                quietly=self._too_soon(event.job_id, "downloading"),
            )
        elif isinstance(event, DownloadStateChangedEvent):
            # Completed and failed are announced by their own events, the
            # failure with its reason.
            if event.state not in ("completed", "failed"):
                state = "downloading" if event.state == "queued" else event.state
                self._record_download(event.job_id, state)
        elif isinstance(event, DownloadCompletedEvent):
            self._record_download(event.job_id, "completed")
        elif isinstance(event, DownloadFailedEvent):
            self._record_download(event.job_id, "failed", error=event.error)

    def _record_download(
        self,
        name: str,
        state: str,
        *,
        bytes_done: int | None = None,
        bytes_total: int | None = None,
        error: str = "",
        quietly: bool = False,
    ) -> None:
        """Keep what the models list shows; tell the shell unless *quietly*."""
        with self._lock:
            previous = self._downloads.get(name, {"bytes_done": 0, "bytes_total": 0})
            download: Json = {
                "name": name,
                "state": state,
                "bytes_done": previous["bytes_done"] if bytes_done is None else bytes_done,
                "bytes_total": previous["bytes_total"] if bytes_total is None else bytes_total,
                "error": error,
            }
            if state in ("downloading", "paused", "failed"):
                self._downloads[name] = download
            else:
                self._downloads.pop(name, None)
            if quietly:
                return
            self._last_progress[name] = (state, self._clock())
        self.publish("download", download)

    def _too_soon(self, key: str, state: str) -> bool:
        with self._lock:
            last = self._last_progress.get(key)
        return (
            last is not None and last[0] == state
            and self._clock() - last[1] < _PROGRESS_SECONDS
        )

    def _on_model_deleted(self, event: DomainEvent) -> None:
        assert isinstance(event, ModelDeletedEvent)
        self._record_download(event.name, "deleted", bytes_done=0, bytes_total=0)

    def _on_microphone_connected(self, event: DomainEvent) -> None:
        assert isinstance(event, MicrophoneConnectedEvent)
        self.publish("microphones", {"name": event.name, "will_be_used": event.will_be_used})

    def _on_history_changed(self, _event: DomainEvent) -> None:
        self.publish("history", {"event": "changed"})

    def _publish_app(self) -> None:
        api = self._api
        if api is not None:
            self.publish("app", api.app_info())

    def _publish_writing(self) -> None:
        api = self._api
        status = None if api is None else api.writing_status()
        if status is None:
            return
        state = str(status["state"])
        if state == "installing" and self._too_soon("writing", state):
            return
        with self._lock:
            self._last_progress["writing"] = (state, self._clock())
        self.publish("writing", status)


def _nothing() -> None:
    return None


def _release_logging_failures(api: Snapshots) -> None:
    try:
        api.release_devices()
    except Exception:  # a failed clean-up must not break the stream's teardown
        _logger.exception("Releasing devices after the last UI client left failed.")
