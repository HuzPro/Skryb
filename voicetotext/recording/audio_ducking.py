"""Lowers or mutes other apps' audio while a recording runs, then puts it back.

Windows keeps a volume per app per playback device (an audio session), the
same one its volume mixer shows. Every session but our own is turned down
when a recording starts or resumes and restored when it pauses or ends;
one the user moves meanwhile is theirs again and stays where they put it.
"""

import logging
import os
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voicetotext.core.event_bus import EventBus
from voicetotext.core.events import (
    DomainEvent,
    RecordingPausedEvent,
    RecordingResumedEvent,
    StateChangedEvent,
)
from voicetotext.core.state_machine import AppState

_logger = logging.getLogger(__name__)

# Apps that start playing mid-recording are caught this often.
POLL_SECONDS = 2.0
# Session volumes come back as 32-bit floats, never quite what was set.
_SAME_VOLUME = 0.005
_QUIT_WAIT_SECONDS = 1.0


@dataclass(frozen=True)
class AudioSession:
    key: str
    pid: int
    volume: Callable[[], float]
    set_volume: Callable[[float], None]


ListSessions = Callable[[], list[AudioSession]]


class SessionDucker:
    """Turns sessions down and back up; call it from one thread only."""

    def __init__(self, list_sessions: ListSessions, own_pid: int) -> None:
        self._list_sessions = list_sessions
        self._own_pid = own_pid
        self._keep: float | None = None
        # Each ducked session's volume before, and the volume we set.
        self._ducked: dict[str, tuple[float, float]] = {}

    @property
    def active(self) -> bool:
        return self._keep is not None

    def duck(self, keep: float) -> None:
        self._keep = keep
        self.duck_new_sessions()

    def duck_new_sessions(self) -> None:
        if self._keep is None:
            return
        for session in self._list_sessions():
            if session.pid != self._own_pid and session.key not in self._ducked:
                self._lower(session, self._keep)

    def restore(self) -> None:
        self._keep = None
        if not self._ducked:
            return
        sessions = self._list_sessions()
        ducked, self._ducked = self._ducked, {}
        for session in sessions:
            if session.key in ducked:
                _restore(session, *ducked[session.key])

    def _lower(self, session: AudioSession, keep: float) -> None:
        try:
            self._ducked[session.key] = _turn_down(session, keep)
        except Exception:  # an app that closed meanwhile must not stop the rest
            _logger.debug("Lowering an app's volume failed.", exc_info=True)


def _turn_down(session: AudioSession, keep: float) -> tuple[float, float]:
    before = session.volume()
    lowered = before * keep
    session.set_volume(lowered)
    return before, lowered


def _restore(session: AudioSession, before: float, lowered: float) -> None:
    try:
        _put_back(session, before, lowered)
    except Exception:  # an app that closed meanwhile must not stop the rest
        _logger.debug("Restoring an app's volume failed.", exc_info=True)


def _put_back(session: AudioSession, before: float, lowered: float) -> None:
    # Moved by the user while lowered: their volume now, not ours to undo.
    if abs(session.volume() - lowered) <= _SAME_VOLUME:
        session.set_volume(before)


class AudioDucking:
    """Drives a :class:`SessionDucker` from the recording's events.

    Core Audio is COM, and a COM call belongs on a thread that has joined
    COM, never the Tk thread; every call here runs in turn on one worker
    thread of its own.
    """

    def __init__(
        self,
        event_bus: EventBus,
        mode: Callable[[], str],
        level: Callable[[], float],
        list_sessions: ListSessions,
        own_pid: int | None = None,
    ) -> None:
        self._mode = mode
        self._level = level
        self._ducker = SessionDucker(
            list_sessions, os.getpid() if own_pid is None else own_pid,
        )
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()
        event_bus.subscribe(StateChangedEvent, self._on_state_changed)
        event_bus.subscribe(RecordingPausedEvent, self._on_paused)
        event_bus.subscribe(RecordingResumedEvent, self._on_resumed)

    def dispose(self) -> None:
        """Put every app's volume back before the process goes."""
        if self._worker is None:
            return
        done = threading.Event()
        self._restore()
        self._send(done.set)
        done.wait(_QUIT_WAIT_SECONDS)

    def _restore(self) -> None:
        # Nothing was ever lowered before the worker's first job.
        if self._worker is not None:
            self._send(self._ducker.restore)

    def _on_state_changed(self, event: DomainEvent) -> None:
        if not isinstance(event, StateChangedEvent):
            return
        if event.new_state == AppState.RECORDING.value:
            self._duck_if_wanted()
        elif event.old_state == AppState.RECORDING.value:
            self._restore()

    def _on_paused(self, _event: DomainEvent) -> None:
        self._restore()

    def _on_resumed(self, _event: DomainEvent) -> None:
        self._duck_if_wanted()

    def _duck_if_wanted(self) -> None:
        mode = self._mode()
        if mode == "off":
            return
        keep = 0.0 if mode == "mute" else min(max(self._level(), 0.0), 1.0)
        self._send(lambda: self._ducker.duck(keep))

    def _send(self, command: Callable[[], None]) -> None:
        with self._worker_lock:
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._serve, daemon=True, name="audio-ducking",
                )
                self._worker.start()
        self._commands.put(command)

    def _serve(self) -> None:
        while True:
            _run_safely(self._next_command())

    def _next_command(self) -> Callable[[], None]:
        try:
            return self._commands.get(
                timeout=POLL_SECONDS if self._ducker.active else None,
            )
        except queue.Empty:
            return self._ducker.duck_new_sessions


def _run_safely(command: Callable[[], None]) -> None:
    try:
        command()
    except Exception:  # the worker must outlive a broken audio stack
        _logger.warning("Changing other apps' volume failed.", exc_info=True)


_com_joined = threading.local()


def core_audio_sessions() -> list[AudioSession]:
    """Every app's audio session on every active playback device."""
    import comtypes
    from pycaw.api.audiopolicy import IAudioSessionManager2
    from pycaw.constants import DEVICE_STATE, EDataFlow
    from pycaw.utils import AudioUtilities

    if not getattr(_com_joined, "done", False):
        comtypes.CoInitializeEx()
        _com_joined.done = True
    endpoints = AudioUtilities.GetDeviceEnumerator().EnumAudioEndpoints(
        EDataFlow.eRender.value, DEVICE_STATE.ACTIVE.value,
    )
    sessions: list[AudioSession] = []
    for index in range(endpoints.GetCount()):
        manager = endpoints.Item(index).Activate(
            IAudioSessionManager2._iid_, comtypes.CLSCTX_ALL, None,
        ).QueryInterface(IAudioSessionManager2)
        listing = manager.GetSessionEnumerator()
        for position in range(listing.GetCount()):
            session = _session(listing.GetSession(position))
            if session is not None:
                sessions.append(session)
    return sessions


def _session(control: Any) -> AudioSession | None:
    try:
        return _describe_session(control)
    except Exception:  # a session that ended while we looked is simply gone
        _logger.debug("Reading an audio session failed.", exc_info=True)
        return None


def _describe_session(control: Any) -> AudioSession:
    from pycaw.api.audioclient import ISimpleAudioVolume
    from pycaw.api.audiopolicy import IAudioSessionControl2

    details = control.QueryInterface(IAudioSessionControl2)
    volume = details.QueryInterface(ISimpleAudioVolume)
    return AudioSession(
        key=str(details.GetSessionInstanceIdentifier()),
        pid=int(details.GetProcessId()),
        volume=lambda: float(volume.GetMasterVolume()),
        set_volume=lambda level: volume.SetMasterVolume(level, None),
    )
