"""One recorded meeting: capture, cut into utterances, transcribe, label, save as it goes."""

import logging
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from skryb.meetings.capture import RATE
from skryb.meetings.models import Labeller
from skryb.meetings.segmenter import Segment, Segmenter
from skryb.meetings.transcript import Publish, Transcript, asr_context

_READ_SECONDS = 0.1
# Quitting closes the audio files and keeps every line already saved, without waiting for the rest.
_QUIT_WAIT_SECONDS = 1.0
_TRACKS = ("me", "them")
_log = logging.getLogger(__name__)


class TrackSource(Protocol):
    def read(self) -> np.ndarray: ...


class MeetingCapture(Protocol):
    them: TrackSource
    me: TrackSource
    output_name: str | None
    warning: str | None

    def start(self) -> None: ...
    def stop(self) -> None: ...


class Models(Protocol):
    def voice_detector(self) -> Labeller: ...
    def speaker_labeller(self) -> tuple[Labeller, str | None]: ...
    def transcribe(self, audio: np.ndarray, context: str | None) -> str: ...
    def release(self) -> None: ...


class _Audio:
    """All audio of one track so far, sliceable by time."""

    def __init__(self) -> None:
        self._data = np.zeros(RATE * 60, np.float32)
        self._length = 0

    def append(self, audio: np.ndarray) -> None:
        if self._length + len(audio) > len(self._data):
            self._data = np.resize(self._data, max(2 * len(self._data), self._length + len(audio)))
        self._data[self._length:self._length + len(audio)] = audio
        self._length += len(audio)

    def between(self, start: float, end: float) -> np.ndarray:
        return self._data[int(start * RATE):min(int(end * RATE), self._length)].copy()


Job = tuple[str, Segment]


class Session:
    """Records until stop(). Audio and transcript land in the folder as they arrive,
    so a crash mid-meeting loses at most the utterance in flight."""

    def __init__(
        self,
        folder: Path,
        started: datetime,
        *,
        capture: MeetingCapture,
        models: Models,
        vocabulary: Path,
        publish: Publish,
        record_microphone: bool = True,
    ) -> None:
        self.folder = folder
        self._publish: Publish = lambda event: publish({**event, "id": folder.name})
        self.transcript = Transcript(folder, started, self._publish)
        self.status = "Starting"
        self.device: str | None = None
        self.warning: str | None = None
        self._capture = capture
        self._models = models
        self._vocabulary = vocabulary
        self._tracks = _TRACKS if record_microphone else ("them",)
        self._audio = {track: _Audio() for track in self._tracks}
        self._files: dict[str, Any] = {}  # soundfile.SoundFile, imported when a meeting starts
        self._inbox: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stopping = threading.Event()
        # The microphone's speech detector loads once the speaker models have: two threads
        # importing torch at once can deadlock. Until then the audio waits in the tracks.
        self._models_loaded = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """Raises when nothing can be recorded, leaving no empty meeting folder behind."""
        self._capture.start()
        try:
            self._open_files()
        except Exception:
            self._capture.stop()
            raise
        self.device = self._capture.output_name
        self._threads = [
            threading.Thread(target=self._record, name="meeting-record", daemon=True),
            threading.Thread(target=self._transcribe, name="meeting-transcribe", daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        self._set_status("Recording")

    def stop(self) -> None:
        """Stop recording, then wait for every utterance already heard to be transcribed."""
        self._set_status("Finishing")
        self._stopping.set()
        for thread in self._threads:
            thread.join()
        self._set_status("Saved")

    def close_quickly(self) -> None:
        """Stop recording and close the audio files, leaving utterances not yet transcribed."""
        self._stopping.set()
        if self._threads:
            self._threads[0].join(_QUIT_WAIT_SECONDS)

    def state(self) -> dict[str, object]:
        return {
            "id": self.folder.name, "status": self.status, "device": self.device, "warning": self.warning,
            **self.transcript.view(),
        }

    def _open_files(self) -> None:
        import soundfile

        self.folder.mkdir(parents=True)
        self._files = {
            track: soundfile.SoundFile(self.folder / f"{track}.flac", "w", RATE, 1, format="FLAC")
            for track in self._tracks
        }

    def _set_status(self, status: str | None = None, warning: str | None = None) -> None:
        """Publish the status; a warning alone keeps the current status."""
        self.status = status or self.status
        if warning:
            self.warning = warning
        self._publish({
            "type": "status", "status": self.status, "folder": str(self.folder),
            "device": self.device, "warning": self.warning,
        })

    def _record(self) -> None:
        try:
            self._record_until_stopped()
        except Exception as exc:
            _log.exception("Meeting recording failed")
            self._set_status(warning=f"Recording stopped: {exc}")
        finally:
            self._capture.stop()
            for file in self._files.values():
                file.close()
            self._inbox.put(("stop", None))

    def _record_until_stopped(self) -> None:
        self._models_loaded.wait()
        voice = self._models.voice_detector() if "me" in self._tracks else None
        mine = Segmenter(voice.frame_seconds if voice is not None else 1.0)
        while True:
            stopping = self._stopping.wait(_READ_SECONDS)
            if voice is not None:
                for segment in mine.push(voice.push(self._take("me"))):
                    self._inbox.put(("me", segment))
            self._inbox.put(("diarize", self._take("them")))
            self._pass_on_capture_news()
            if stopping:
                break
        for segment in mine.flush():
            self._inbox.put(("me", segment))

    def _pass_on_capture_news(self) -> None:
        if self._capture.warning:
            self._set_status(warning=self._capture.warning)
            self._capture.warning = None
        if self._capture.output_name != self.device:
            self.device = self._capture.output_name
            self._set_status()

    def _take(self, track: str) -> np.ndarray:
        audio = getattr(self._capture, track).read()
        self._files[track].write(audio)
        self._audio[track].append(audio)
        return audio

    def _transcribe(self) -> None:
        try:
            self._run_models()
        except Exception as exc:
            _log.exception("Meeting transcription failed")
            self._set_status(warning=f"Transcription stopped: {exc}. Audio is still being saved.")
            self._models_loaded.set()
            self._drain_until_stop()
        finally:
            self._models_loaded.set()
            self._models.release()

    def _drain_until_stop(self) -> None:
        while self._inbox.get()[0] != "stop":
            pass

    def _run_models(self) -> None:
        self._set_status("Loading models")
        labeller, warning = self._models.speaker_labeller()
        self._models_loaded.set()
        self._set_status("Finishing" if self._stopping.is_set() else "Recording", warning)
        theirs = Segmenter(labeller.frame_seconds)
        jobs: list[Job] = []
        done = False
        while not done or jobs:
            for kind, payload in self._drain(block=not jobs and not done):
                if kind == "diarize":
                    assert isinstance(payload, np.ndarray)
                    jobs += [("them", s) for s in theirs.push(labeller.push(payload))]
                elif kind == "me":
                    assert isinstance(payload, Segment)
                    jobs.append(("me", payload))
                elif kind == "stop":
                    jobs += [("them", s) for s in theirs.push(labeller.push(np.zeros(0, np.float32), last=True))]
                    jobs += [("them", s) for s in theirs.flush()]
                    done = True
            if jobs:
                self._transcribe_one(*jobs.pop(0))

    def _transcribe_one(self, track: str, segment: Segment) -> None:
        audio = self._audio[track].between(segment.start, segment.end)
        if not len(audio):
            return
        context = asr_context(self._vocabulary, dict(self.transcript.names), segment)
        self.transcript.add(track, segment, self._models.transcribe(audio, context))

    def _drain(self, block: bool) -> list[tuple[str, object]]:
        items = [self._inbox.get()] if block else []
        while True:
            try:
                items.append(self._inbox.get_nowait())
            except queue.Empty:
                return items
