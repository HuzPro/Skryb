import logging
import math
from collections.abc import Callable

import numpy as np

from voicetotext.recording.recording_cue_player import (
    SAMPLE_RATE,
    START_FREQUENCY_HZ,
    STOP_FREQUENCY_HZ,
    ReadyCuePlayer,
    RecordingCuePlayer,
    ready_chime_to_samples,
    tone_to_samples,
    warning_to_samples,
)
from voicetotext.recording.preferences import RecordingPreferences


_logger = logging.getLogger(__name__)

# The volume setting's default plays the cues exactly as loud as they were
# before there was a setting; the top of the slider is a little louder.
_UNCHANGED_VOLUME = RecordingPreferences.cue_volume


def _on() -> bool:
    return True


def _unchanged() -> float:
    return _UNCHANGED_VOLUME


class SoundDeviceCuePlayer(RecordingCuePlayer, ReadyCuePlayer):
    """Plays cues on the default output device via sounddevice (non-blocking).

    Cue playback must never break recording, so a missing or busy output
    device is swallowed (logged at debug) rather than raised. Whether to
    play and how loud are asked at every cue, so the settings apply live.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        enabled: Callable[[], bool] = _on,
        volume: Callable[[], float] = _unchanged,
    ) -> None:
        self._sample_rate = sample_rate
        self._enabled = enabled
        self._volume = volume
        self._start = tone_to_samples(START_FREQUENCY_HZ)
        self._stop = tone_to_samples(STOP_FREQUENCY_HZ)
        self._ready = ready_chime_to_samples()
        self._warning = warning_to_samples()

    def play_start(self) -> None:
        self._play(self._start)

    def play_stop(self) -> None:
        self._play(self._stop)

    def play_ready(self) -> None:
        self._play(self._ready)

    def play_warning(self) -> None:
        self._play(self._warning)

    def _play(self, samples: np.ndarray) -> None:
        if not self._enabled():
            return
        gain = min(max(self._volume(), 0.0), 1.0) / _UNCHANGED_VOLUME
        if not math.isfinite(gain) or gain <= 0:
            return
        try:
            self._emit(samples * np.float32(gain))
        except Exception:  # noqa: BLE001 - a cue must never break recording
            _logger.debug("recording cue playback failed", exc_info=True)

    def _emit(self, samples: np.ndarray) -> None:
        import sounddevice as sd

        sd.play(samples, self._sample_rate, blocking=False)
