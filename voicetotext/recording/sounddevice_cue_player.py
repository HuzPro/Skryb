import logging

import numpy as np

from voicetotext.recording.recording_cue_player import (
    SAMPLE_RATE,
    START_FREQUENCY_HZ,
    STOP_FREQUENCY_HZ,
    ReadyCuePlayer,
    RecordingCuePlayer,
    ready_chime_to_samples,
    tone_to_samples,
)


_logger = logging.getLogger(__name__)


class SoundDeviceCuePlayer(RecordingCuePlayer, ReadyCuePlayer):
    """Plays cues on the default output device via sounddevice (non-blocking).

    Cue playback must never break recording, so a missing or busy output
    device is swallowed (logged at debug) rather than raised.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._sample_rate = sample_rate
        self._start = tone_to_samples(START_FREQUENCY_HZ)
        self._stop = tone_to_samples(STOP_FREQUENCY_HZ)
        self._ready = ready_chime_to_samples()

    def play_start(self) -> None:
        self._play(self._start)

    def play_stop(self) -> None:
        self._play(self._stop)

    def play_ready(self) -> None:
        self._play(self._ready)

    def _play(self, samples: np.ndarray) -> None:
        try:
            self._emit(samples)
        except Exception:  # noqa: BLE001 - a cue must never break recording
            _logger.debug("recording cue playback failed", exc_info=True)

    def _emit(self, samples: np.ndarray) -> None:
        import sounddevice as sd

        sd.play(samples, self._sample_rate, blocking=False)
