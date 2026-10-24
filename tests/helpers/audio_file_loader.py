"""Decode an audio file on disk into an :class:`AudioSegment`.

This is a thin test-only bridge between ``soundfile`` and the
``TranscriptionEngine`` port.  Production code does not need it today;
if a "transcribe a file" feature ever lands, promote this module to
``skryb/recording/`` behind an ``AudioFileLoader`` Protocol.

The loader always returns mono float32 samples at 16 kHz, matching the
application's transcription input format.
"""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile
from scipy.signal import resample_poly

from skryb.recording.audio_segment import AudioSegment


_TARGET_SAMPLE_RATE = 16000


def load_audio_file(path: Path) -> AudioSegment:
    if not path.exists():
        raise FileNotFoundError(path)
    data, source_rate = soundfile.read(
        str(path), dtype="float32", always_2d=False
    )
    data = _downmix_to_mono(data)
    data = _resample_to_target(data, source_rate)
    return AudioSegment(data=data, sample_rate=_TARGET_SAMPLE_RATE)


def _downmix_to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.mean(axis=1).astype(np.float32)


def _resample_to_target(data: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == _TARGET_SAMPLE_RATE:
        return data
    divisor = gcd(source_rate, _TARGET_SAMPLE_RATE)
    up = _TARGET_SAMPLE_RATE // divisor
    down = source_rate // divisor
    resampled = resample_poly(data, up=up, down=down)
    return resampled.astype(np.float32)
