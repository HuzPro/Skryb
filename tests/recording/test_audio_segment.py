import numpy as np
import pytest

from voicetotext.recording.audio_segment import AudioSegment


class TestAudioSegmentCreation:
    def test_should_create_segment_from_valid_float32_audio(self):
        data = np.zeros(16000, dtype=np.float32)

        segment = AudioSegment(data=data, sample_rate=16000)

        assert segment.sample_rate == 16000
        assert segment.data.dtype == np.float32

    def test_should_calculate_duration_in_seconds(self):
        data = np.zeros(32000, dtype=np.float32)

        segment = AudioSegment(data=data, sample_rate=16000)

        assert segment.duration_seconds == 2.0

    def test_should_report_sample_count(self):
        data = np.zeros(8000, dtype=np.float32)

        segment = AudioSegment(data=data, sample_rate=16000)

        assert segment.sample_count == 8000


class TestAudioSegmentValidation:
    def test_should_reject_non_float32_dtype(self):
        data = np.zeros(16000, dtype=np.int16)

        with pytest.raises(ValueError, match="float32"):
            AudioSegment(data=data, sample_rate=16000)

    def test_should_reject_zero_sample_rate(self):
        data = np.zeros(16000, dtype=np.float32)

        with pytest.raises(ValueError, match="sample_rate"):
            AudioSegment(data=data, sample_rate=0)

    def test_should_reject_negative_sample_rate(self):
        data = np.zeros(16000, dtype=np.float32)

        with pytest.raises(ValueError, match="sample_rate"):
            AudioSegment(data=data, sample_rate=-1)

    def test_should_reject_empty_audio_data(self):
        data = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="empty"):
            AudioSegment(data=data, sample_rate=16000)


class TestAudioSegmentImmutability:
    def test_should_be_immutable(self):
        data = np.zeros(16000, dtype=np.float32)
        segment = AudioSegment(data=data, sample_rate=16000)

        with pytest.raises(AttributeError):
            segment.sample_rate = 44100
