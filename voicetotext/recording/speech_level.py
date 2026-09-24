import logging

import numpy as np

from voicetotext.recording.audio_segment import AudioSegment

_logger = logging.getLogger(__name__)

# Raise quiet speech until its loudest sample reaches half of full scale.
# Scaling by the true peak is what makes clipping impossible: no sample
# can land above the target.
_TARGET_PEAK = 0.5
# 20 dB. Past this the room noise comes up faster than the voice helps.
_MAX_GAIN = 10.0
# A recording whose loudest moment is this quiet holds no speech worth
# lifting, only the noise floor.
_SILENCE_PEAK = 0.01
_SMALLEST_USEFUL_GAIN = 1.05


def boost_quiet_speech(audio: AudioSegment) -> AudioSegment:
    """Bring speech recorded far from the microphone up to a normal level."""
    peak = float(np.max(np.abs(audio.data)))
    if peak < _SILENCE_PEAK:
        return audio
    gain = min(_TARGET_PEAK / peak, _MAX_GAIN)
    if gain < _SMALLEST_USEFUL_GAIN:
        return audio
    _logger.info("Boosting quiet speech x%.1f (peak %.3f).", gain, peak)
    return AudioSegment(
        data=(audio.data * np.float32(gain)).astype(np.float32),
        sample_rate=audio.sample_rate,
    )
