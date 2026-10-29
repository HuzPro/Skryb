"""Starts and stops meeting recordings, from the tray or the window, one at a time."""

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from skryb.meetings.session import MeetingCapture, Models, Session
from skryb.meetings.storage import MeetingSettings, new_meeting_folder, vocabulary_file
from skryb.meetings.transcript import Event

_logger = logging.getLogger(__name__)


class MeetingError(Exception):
    """A meeting that could not start or stop; the message is a sentence to show as is."""


def _nothing(_recording: bool) -> None:
    return None


class MeetingRecorder:
    def __init__(
        self,
        *,
        settings: Callable[[], MeetingSettings],
        missing_models: Callable[[], list[str]],
        make_models: Callable[[MeetingSettings], Models],
        make_capture: Callable[[bool], MeetingCapture],
        publish: Callable[[Event], None],
        on_change: Callable[[bool], None] = _nothing,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._settings = settings
        self._missing_models = missing_models
        self._make_models = make_models
        self._make_capture = make_capture
        self._publish = publish
        self._on_change = on_change
        self._clock = clock
        self._turn = threading.Lock()  # one start or stop at a time
        self._current: Session | None = None
        self._last: Session | None = None

    def is_recording(self) -> bool:
        return self._current is not None

    def recording_started(self) -> datetime | None:
        current = self._current
        return None if current is None else current.transcript.started

    def live(self) -> dict[str, object] | None:
        """The meeting being recorded or still finishing its transcript; none once it is saved."""
        session = self._current or self._last
        if session is None or (session is self._last and session.status == "Saved"):
            return None
        return {"recording": session is self._current, **session.state()}

    def live_session(self, meeting_id: str) -> Session | None:
        session = self._current or self._last
        return session if session is not None and session.folder.name == meeting_id else None

    def start(self) -> None:
        with self._turn:
            if self._current is not None:
                return
            missing = self._missing_models()
            if missing:
                raise MeetingError(f"Download the meeting models first: {', '.join(missing)}.")
            settings = self._settings()
            started = self._clock()
            session = Session(
                new_meeting_folder(settings.folder, started), started,
                capture=self._make_capture(settings.record_microphone),
                models=self._make_models(settings),
                vocabulary=self._vocabulary(settings.folder),
                publish=self._publish,
                record_microphone=settings.record_microphone,
            )
            try:
                session.start()
            except Exception as exc:
                _logger.exception("Could not start a meeting recording")
                raise MeetingError(f"Could not start recording: {exc}") from exc
            self._current = session
            _logger.info("Meeting recording started in %s", session.folder)
        self._publish({"type": "started", **session.state()})
        self._on_change(True)

    def stop(self) -> None:
        with self._turn:
            session, self._current = self._current, None
            if session is None:
                return
            self._last = session
            self._on_change(False)
            session.stop()
            _logger.info("Meeting saved in %s", session.folder)
        self._publish({"type": "saved", "id": session.folder.name})

    def dispose(self) -> None:
        """On quit: the audio files close and the lines so far stay saved; the rest is not transcribed."""
        session = self._current
        if session is not None:
            session.close_quickly()

    def toggle(self) -> None:
        if self.is_recording():
            self.stop()
        else:
            self.start()

    @staticmethod
    def _vocabulary(folder: Path) -> Path:
        try:
            return vocabulary_file(folder)
        except OSError:
            return folder / "vocabulary.txt"
