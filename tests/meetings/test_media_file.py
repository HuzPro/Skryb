"""Reading sound out of real files through Windows' Media Foundation. Windows only: CI on Linux skips these."""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile

from skryb.meetings.capture import RATE

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Media Foundation is part of Windows")

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "media" / "tone-440hz.mp4"


def _peak_hz(audio: np.ndarray) -> float:
    window = audio[:RATE]
    return float(np.fft.rfftfreq(len(window), 1 / RATE)[np.abs(np.fft.rfft(window)).argmax()])


def test_a_video_s_sound_comes_out_as_16_khz_mono() -> None:
    from skryb.meetings.media_file import decode_audio

    audio = decode_audio(_FIXTURE)

    assert audio.dtype == np.float32
    assert abs(len(audio) / RATE - 2.0) < 0.1
    assert abs(_peak_hz(audio) - 440) < 2


def test_a_stereo_wav_at_another_rate_is_mixed_down_and_resampled(tmp_path: Path) -> None:
    from skryb.meetings.media_file import decode_audio

    t = np.arange(44100 * 3) / 44100
    stereo = np.stack([0.3 * np.sin(2 * np.pi * 300 * t), 0.3 * np.sin(2 * np.pi * 300 * t)], axis=1)
    path = tmp_path / "تقریر stereo.wav"
    soundfile.write(path, stereo, 44100)
    shares: list[float] = []

    audio = decode_audio(path, progress=shares.append)

    assert len(audio) == 3 * RATE
    assert abs(_peak_hz(audio) - 300) < 2
    assert shares and shares == sorted(shares) and shares[-1] <= 1.0


def test_reading_stops_once_cancelled_and_keeps_what_was_read(tmp_path: Path) -> None:
    from skryb.meetings.media_file import decode_audio

    path = tmp_path / "long.wav"
    soundfile.write(path, np.zeros(RATE * 60, np.float32), RATE)
    blocks = 0

    def cancelled() -> bool:
        nonlocal blocks
        blocks += 1
        return blocks > 3

    assert len(decode_audio(path, cancelled=cancelled)) < RATE * 60


@pytest.mark.parametrize(("content", "message"), [
    (b"this is not a video at all", "Skryb can't read .mp4 files yet. It reads MP4, MOV, MKV, WebM"),
    (None, "isn't there any more."),
])
def test_a_file_windows_cant_read_is_refused_with_a_sentence(tmp_path: Path, content: bytes | None, message: str) -> None:
    from skryb.meetings.media_file import MediaFileError, check

    path = tmp_path / "meeting.mp4"
    if content is not None:
        path.write_bytes(content)

    with pytest.raises(MediaFileError, match=message):
        check(path)
