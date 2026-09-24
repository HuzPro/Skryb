import wave

import numpy as np
import pytest

from voicetotext.recording.audio_segment import AudioSegment, load_wav


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


def _write_wav(path, samples, rate=16000, channels=1, width=2) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())


class TestLoadWav:
    def test_reads_a_saved_recording_back_as_float_mono(self, tmp_path):
        pcm = np.array([0, 16384, -16384, 32767, -32768], dtype="<i2")
        _write_wav(tmp_path / "0001.wav", pcm)

        audio = load_wav(tmp_path / "0001.wav")

        assert audio.sample_rate == 16000
        assert audio.data.dtype == np.float32
        np.testing.assert_allclose(
            audio.data, [0.0, 0.5, -0.5, 32767 / 32768, -1.0], atol=1e-6,
        )

    def test_mixes_stereo_down_to_mono(self, tmp_path):
        pcm = np.array([16384, 0, -16384, -16384], dtype="<i2")
        _write_wav(tmp_path / "stereo.wav", pcm, rate=44100, channels=2)

        audio = load_wav(tmp_path / "stereo.wav")

        np.testing.assert_allclose(audio.data, [0.25, -0.5], atol=1e-6)
        assert audio.sample_rate == 44100

    def test_reads_ten_minutes_of_audio(self, tmp_path):
        pcm = (np.sin(np.arange(16000 * 600) / 20) * 20000).astype("<i2")
        _write_wav(tmp_path / "long.wav", pcm)

        audio = load_wav(tmp_path / "long.wav")

        assert audio.duration_seconds == pytest.approx(600.0)

    def test_refuses_formats_the_app_never_writes(self, tmp_path):
        _write_wav(tmp_path / "8bit.wav", np.array([1, 2, 3], dtype="u1"), width=1)

        with pytest.raises(wave.Error):
            load_wav(tmp_path / "8bit.wav")

    def test_refuses_an_empty_recording(self, tmp_path):
        _write_wav(tmp_path / "empty.wav", np.array([], dtype="<i2"))

        with pytest.raises(ValueError):
            load_wav(tmp_path / "empty.wav")
