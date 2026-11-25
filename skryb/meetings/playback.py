"""A stretch of a meeting to play back: every track mixed, as the meeting sounded."""

import io
from pathlib import Path

import numpy as np

from skryb.meetings.capture import RATE

TRACKS = ("me", "them")


def mixed(tracks: list[np.ndarray]) -> np.ndarray:
    """The tracks played together; a shorter one simply ends early."""
    together = np.zeros(max((len(track) for track in tracks), default=0), np.float32)
    for track in tracks:
        together[:len(track)] += track
    return np.clip(together, -1.0, 1.0)


def read_mixed(folder: Path, start: float, end: float) -> np.ndarray:
    """*start* to *end* seconds of the saved tracks in *folder*, mixed; FileNotFoundError when it has none."""
    import soundfile

    paths = [folder / f"{track}.flac" for track in TRACKS if (folder / f"{track}.flac").is_file()]
    if not paths:
        raise FileNotFoundError(f"{folder} has no recording.")
    tracks = []
    for path in paths:
        with soundfile.SoundFile(path) as sound:
            sound.seek(min(int(start * sound.samplerate), sound.frames))
            audio = sound.read(max(0, int((end - start) * sound.samplerate)), dtype="float32")
        tracks.append(audio.mean(axis=1) if audio.ndim > 1 else audio)
    return mixed(tracks)


def wav(audio: np.ndarray) -> bytes:
    import soundfile

    buffer = io.BytesIO()
    soundfile.write(buffer, audio, RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
