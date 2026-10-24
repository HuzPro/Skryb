import numpy as np

from skryb.recording.audio_segment import AudioSegment


class AudioSegmentBuilder:
    """Fluent builder for :class:`AudioSegment`.

    Starts with a sensible default (1 second of silence at 16kHz) and
    lets tests override each facet independently.  Every method
    returns ``self`` so calls can be chained.
    """

    def __init__(self) -> None:
        self._sample_rate: int = 16000
        self._duration_seconds: float = 1.0
        self._fill_value: float = 0.0

    def with_duration(self, seconds: float) -> "AudioSegmentBuilder":
        self._duration_seconds = seconds
        return self

    def at_sample_rate(self, sample_rate: int) -> "AudioSegmentBuilder":
        self._sample_rate = sample_rate
        return self

    def silent(self) -> "AudioSegmentBuilder":
        self._fill_value = 0.0
        return self

    def filled_with(self, value: float) -> "AudioSegmentBuilder":
        self._fill_value = value
        return self

    def build(self) -> AudioSegment:
        sample_count = max(1, int(self._sample_rate * self._duration_seconds))
        data = np.full(sample_count, self._fill_value, dtype=np.float32)
        return AudioSegment(data=data, sample_rate=self._sample_rate)


def an_audio_segment() -> AudioSegmentBuilder:
    return AudioSegmentBuilder()
