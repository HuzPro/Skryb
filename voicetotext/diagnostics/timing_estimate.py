import json
import logging
import math
import threading
from collections import deque
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

_SAMPLES_PER_ENGINE = 200
_MIN_SAMPLES = 3
_MIN_ESTIMATE_SECONDS = 0.05
_SESSIONS_TO_SEED_FROM = 100


class TranscriptionTimeEstimate:
    """How long a transcription should take, learnt from how long past ones did.

    Kept per model, engine and device, since a model on the CPU and the same
    model on the GPU are different machines as far as the wait goes. The
    fit is a straight line through (audio length, wall time), because a
    fixed start-up cost dominates short clips and the audio length
    dominates long ones.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._samples: dict[str, deque[tuple[float, float]]] = {}

    def record(self, key: str, audio_seconds: float, wall_seconds: float) -> None:
        if not (_is_positive(audio_seconds) and _is_positive(wall_seconds)):
            return
        with self._lock:
            self._samples.setdefault(
                key, deque(maxlen=_SAMPLES_PER_ENGINE),
            ).append((audio_seconds, wall_seconds))

    def estimate(self, key: str, audio_seconds: float) -> float | None:
        with self._lock:
            samples = list(self._samples.get(key, ()))
        if len(samples) < _MIN_SAMPLES or not _is_positive(audio_seconds):
            return None
        audio = np.array([sample[0] for sample in samples])
        wall = np.array([sample[1] for sample in samples])
        # ponytail: least squares, so a slow first run after a model load
        # skews it for a while. Switch to Theil-Sen if the bar visibly lies.
        if np.ptp(audio) > 0:
            slope, intercept = np.polyfit(audio, wall, 1)
            if slope >= 0:
                return max(float(intercept + slope * audio_seconds), _MIN_ESTIMATE_SECONDS)
        return max(float(np.median(wall)), _MIN_ESTIMATE_SECONDS)

    def seed_from(self, diagnostics_dir: Path, skip: Path) -> None:
        """Load past timings, oldest first, from the newest session folders."""
        try:
            sessions = sorted(
                (path for path in diagnostics_dir.iterdir() if path != skip),
                reverse=True,
            )[:_SESSIONS_TO_SEED_FROM]
        except OSError as exc:
            _logger.debug("No past timings to learn from: %s", exc)
            return
        for session in reversed(sessions):
            for event in _transcription_events(session / "session.jsonl"):
                self.record(
                    engine_key(event.get("model"), event.get("engine"), event.get("device")),
                    _number(event.get("audio_seconds")),
                    _number(event.get("wall_seconds"))
                    - _number(event.get("output_seconds")),
                )


def engine_key(model: object, engine: object, device: object) -> str:
    return f"{model}|{engine}|{device}"


def _transcription_events(path: Path) -> list[dict[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if (
            isinstance(event, dict)
            and event.get("event") == "transcription"
            and not event.get("error")
        ):
            events.append(event)
    return events


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _is_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0
