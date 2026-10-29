"""Turns per-frame speaker activity into utterances to transcribe."""

from collections.abc import Iterable
from dataclasses import dataclass

SILENCE = -1


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    speaker: int


class Segmenter:
    """Feeds on one label per frame (a speaker index, or SILENCE) and closes an
    utterance on a pause, on a speaker change that sticks, or at a length cap
    that keeps text arriving during long monologues."""

    def __init__(
        self, frame_seconds: float, min_silence: float = 0.5, min_turn: float = 0.3,
        max_length: float = 15.0, min_length: float = 0.3,
    ) -> None:
        self._frame = frame_seconds
        self._min_silence = round(min_silence / frame_seconds)
        self._min_turn = round(min_turn / frame_seconds)
        self._max_length = round(max_length / frame_seconds)
        self._min_length = round(min_length / frame_seconds)
        self._i = 0
        self._start: int | None = None
        self._speaker = 0
        self._last_speech = 0
        # (speaker, first frame, frames seen) of a possible turn change
        self._pending: tuple[int, int, int] | None = None

    def push(self, labels: Iterable[int]) -> list[Segment]:
        done: list[Segment] = []
        for label in labels:
            self._step(int(label), done)
            self._i += 1
        return done

    def flush(self) -> list[Segment]:
        done: list[Segment] = []
        if self._start is not None:
            self._close(self._last_speech + 1, done)
        return done

    def _step(self, label: int, done: list[Segment]) -> None:
        i = self._i
        if label == SILENCE:
            self._pending = None
            if self._start is not None and i - self._last_speech >= self._min_silence:
                self._close(self._last_speech + 1, done)
            return
        if self._start is None:
            self._start, self._speaker, self._last_speech = i, label, i
            return
        self._last_speech = i
        if label == self._speaker:
            self._pending = None
        elif self._pending and self._pending[0] == label:
            speaker, first, seen = self._pending
            self._pending = (speaker, first, seen + 1)
            if seen + 1 >= self._min_turn:
                self._close(first, done)
                self._start, self._speaker = first, speaker
        else:
            self._pending = (label, i, 1)
        if self._start is not None and i + 1 - self._start >= self._max_length:
            self._close(i + 1, done)

    def _close(self, end: int, done: list[Segment]) -> None:
        assert self._start is not None
        if end - self._start >= self._min_length:
            done.append(Segment(self._start * self._frame, end * self._frame, self._speaker))
        self._start = None
        self._pending = None
