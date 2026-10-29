"""Stand-ins for the sound devices and the meeting models, so meetings run on the CPU in tests.

Audio is synthetic: a speaker's voice is a constant level that names them
(0.2 is the first voice, 0.4 the second, 0.6 the third), and silence is zero.
"""

import threading
import time

import numpy as np

from skryb.meetings.capture import RATE
from skryb.meetings.segmenter import SILENCE

_VOICE_LEVELS = (0.2, 0.4, 0.6)
# What each voice says, so the mic and the speakers never read as an echo of each other.
PHRASES = ("All right, let's jump in.", "Sounds good, I pulled the numbers from QuoteDesk.", "Haan theek hai, main kal bhej dunga.")


def voice(speaker: int, seconds: float) -> np.ndarray:
    return np.full(round(seconds * RATE), _VOICE_LEVELS[speaker], np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * RATE), np.float32)


class ScriptedTrack:
    """Hands out its audio a second per read, then silence; ``done`` is set once it has all gone."""

    def __init__(self, audio: np.ndarray) -> None:
        self._audio = audio
        self.done = threading.Event()

    def read(self) -> np.ndarray:
        piece, self._audio = self._audio[:RATE], self._audio[RATE:]
        if not len(piece):
            self.done.set()
        return piece


class FakeCapture:
    def __init__(self, them: np.ndarray, me: np.ndarray | None = None, *, fails: Exception | None = None) -> None:
        self.them = ScriptedTrack(them)
        self.me = ScriptedTrack(silence(0) if me is None else me)
        self.output_name: str | None = "Headset Earphone (Jabra Evolve2 65)"
        self.warning: str | None = None
        self.started = False
        self.stopped = threading.Event()
        self._fails = fails

    def start(self) -> None:
        if self._fails is not None:
            raise self._fails
        self.started = True

    def stop(self) -> None:
        self.stopped.set()


class LevelLabeller:
    """Labels each frame by its level: silence, or the voice that level names (always 0 for speech detection)."""

    def __init__(self, frame_seconds: float, speech_only: bool = False) -> None:
        self.frame_seconds = frame_seconds
        self._speech_only = speech_only
        self._frame = round(frame_seconds * RATE)
        self._rest = np.zeros(0, np.float32)
        self.finished = False

    def push(self, audio: np.ndarray, last: bool = False) -> list[int]:
        audio = np.concatenate([self._rest, audio])
        whole = len(audio) // self._frame * self._frame
        self._rest = audio[whole:]
        self.finished = self.finished or last
        labels = [_voice_of(float(np.abs(frame).mean())) for frame in audio[:whole].reshape(-1, self._frame)]
        return [label if label == SILENCE or not self._speech_only else 0 for label in labels]


def _voice_of(level: float) -> int:
    if level < 0.1:
        return SILENCE
    return min(range(len(_VOICE_LEVELS)), key=lambda speaker: abs(_VOICE_LEVELS[speaker] - level))


class FakeModels:
    """Writes what each clip's voice says, or what ``texts`` says, in order."""

    def __init__(self, texts: list[str] | None = None, *, fails: Exception | None = None, warning: str | None = None) -> None:
        self.texts = list(texts or [])
        self.contexts: list[str | None] = []
        self.voice_asked = 0
        self.released = 0
        self.speakers: LevelLabeller | None = None
        self.loaded: list[str] = []
        self._fails = fails
        self._warning = warning

    def voice_detector(self) -> LevelLabeller:
        self.voice_asked += 1
        self.loaded.append("speech detection")
        return LevelLabeller(0.032, speech_only=True)

    def speaker_labeller(self) -> tuple[LevelLabeller, str | None]:
        if self._fails is not None:
            raise self._fails
        time.sleep(0.2)  # loading takes a while; the microphone must wait for it
        self.loaded.append("speaker labels")
        self.speakers = LevelLabeller(0.01)
        return self.speakers, self._warning

    def transcribe(self, audio: np.ndarray, context: str | None) -> str:
        self.contexts.append(context)
        if self.texts:
            return self.texts.pop(0)
        return PHRASES[_voice_of(float(np.abs(audio).mean()))]

    def release(self) -> None:
        self.released += 1
