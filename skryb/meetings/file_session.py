"""A video or audio file transcribed as a meeting: read, told apart by speaker, transcribed, saved as it goes.

It has the surface of a recording ``Session`` (folder, transcript, status, ``state``, ``stop``,
``close_quickly``), so the recorder and the Meetings window treat it as the meeting in progress.
"""

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from skryb.meetings.capture import RATE
from skryb.meetings.media_file import MediaFileError, decode_audio
from skryb.meetings.segmenter import Segment, Segmenter
from skryb.meetings.session import Models
from skryb.meetings.transcript import Publish, Transcript, asr_context

# The speaker labeller hears the file in pieces this long, as it would hear a live meeting.
_PIECE_SECONDS = 10
# Quitting keeps every line already saved without waiting for the rest.
_QUIT_WAIT_SECONDS = 1.0
# Progress is published in steps of a percent, not for every block read.
_PROGRESS_STEP = 0.01
_log = logging.getLogger(__name__)

Decode = Callable[..., np.ndarray]


class FileSession:
    """Transcribes *source* into *folder* on its own thread; ``stop`` cancels and keeps the lines so far."""

    def __init__(
        self,
        folder: Path,
        started: datetime,
        *,
        source: Path,
        models: Models,
        vocabulary: Path,
        publish: Publish,
        on_done: Callable[["FileSession"], None],
        decode: Decode = decode_audio,
    ) -> None:
        self.folder = folder
        self.source = source
        self._publish: Publish = lambda event: publish({**event, "id": folder.name})
        self.transcript = Transcript(folder, started, self._publish)
        self.status = "Starting"
        self.device: str | None = None
        self.warning: str | None = None
        self.progress: float | None = None
        self._published: tuple[str, float | None] = ("", None)
        self._models = models
        self._vocabulary = vocabulary
        self._on_done = on_done
        self._decode = decode
        self._cancel = threading.Event()
        self._thread = threading.Thread(target=self._run, name="meeting-file", daemon=True)

    def start(self) -> None:
        """Create the meeting folder; nothing is read or published until ``begin``."""
        self.folder.mkdir(parents=True)
        self.transcript.save()  # listed with the other meetings from the start, even if nothing is said
        self.status, self.progress = "Reading the file", 0.0

    def begin(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Cancel: stop after the line being transcribed, keeping every line already saved."""
        self._cancel.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join()

    def close_quickly(self) -> None:
        self._cancel.set()
        self._thread.join(_QUIT_WAIT_SECONDS)

    def state(self) -> dict[str, object]:
        return {
            "id": self.folder.name, "status": self.status, "device": self.device, "warning": self.warning,
            "source": self.source.name, "progress": self.progress, **self.transcript.view(),
        }

    def _set_status(self, status: str, warning: str | None = None, progress: float | None = None) -> None:
        self.status = status
        self.progress = progress
        if warning:
            self.warning = warning
        last_status, last_progress = self._published
        if (not warning and status == last_status and progress is not None and last_progress is not None
                and progress < 1.0 and progress - last_progress < _PROGRESS_STEP):
            return
        self._published = (status, progress)
        self._publish({
            "type": "status", "status": self.status, "folder": str(self.folder), "device": self.device,
            "warning": self.warning, "source": self.source.name, "progress": self.progress,
        })

    def _run(self) -> None:
        try:
            self._transcribe_file()
        except MediaFileError as exc:
            self.warning = str(exc)
        except Exception as exc:
            _log.exception("Transcribing %s failed", self.source)
            self.warning = f"Transcription stopped: {exc}"
        finally:
            self._models.release()
            if self._cancel.is_set() and not self.warning:
                self.warning = "Cancelled: the lines so far are saved."
            self._set_status("Saved")
            self._on_done(self)

    def _transcribe_file(self) -> None:
        audio = self._decode(
            self.source,
            progress=lambda share: self._set_status("Reading the file", progress=round(share, 3)),
            cancelled=self._cancel.is_set,
        )
        self._save_audio(audio)
        if self._cancel.is_set():
            return
        if not len(audio):
            raise MediaFileError(f"{self.source.name} has no sound to transcribe.")
        self._set_status("Loading models")
        labeller, warning = self._models.speaker_labeller()
        self._set_status("Transcribing", warning, progress=0.0)
        segmenter = Segmenter(labeller.frame_seconds)
        seconds = len(audio) / RATE
        step = _PIECE_SECONDS * RATE
        for start in range(0, len(audio), step):
            last = start + step >= len(audio)
            segments = segmenter.push(labeller.push(audio[start:start + step], last=last))
            if last:
                segments += segmenter.flush()
            for segment in segments:
                if self._cancel.is_set():
                    return
                self._transcribe_one(audio, segment)
                self._set_status("Transcribing", progress=round(min(1.0, segment.end / seconds), 3))
        self._set_status("Transcribing", progress=1.0)

    def _transcribe_one(self, audio: np.ndarray, segment: Segment) -> None:
        piece = audio[int(segment.start * RATE):int(segment.end * RATE)]
        if not len(piece):
            return
        context = asr_context(self._vocabulary, dict(self.transcript.names), segment)
        self.transcript.add("them", segment, self._models.transcribe(piece, context))

    def _save_audio(self, audio: np.ndarray) -> None:
        """The file's sound beside the transcript, as a recorded meeting keeps its own; the video is not copied."""
        import soundfile

        soundfile.write(self.folder / "them.flac", audio, RATE, format="FLAC")
