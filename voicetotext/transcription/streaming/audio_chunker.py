from collections.abc import Iterator

import numpy as np

from voicetotext.recording.audio_segment import AudioSegment


def iter_chunks(
    audio: AudioSegment, chunk_seconds: float,
) -> Iterator[np.ndarray]:
    """Slice *audio* into fixed-length frames, the way a mic delivers them.

    The final frame may be shorter than the rest. Feeding a fixture
    through this generator lets streaming tests exercise the engine with
    the same incremental shape the live recorder will eventually push.
    """
    chunk_samples = max(1, int(chunk_seconds * audio.sample_rate))
    data = audio.data
    for start in range(0, len(data), chunk_samples):
        yield data[start:start + chunk_samples]
