import numpy as np
import pytest

from skryb.recording.audio_segment import AudioSegment
from skryb.recording.speech_level import boost_quiet_speech


def _speech(peak: float, seconds: float = 2.0, noise: float = 0.0) -> AudioSegment:
    rate = 16000
    time = np.arange(int(rate * seconds)) / rate
    voice = np.sin(2 * np.pi * 220 * time) * peak
    hiss = np.random.default_rng(7).normal(0, noise, time.size) if noise else 0
    return AudioSegment(data=(voice + hiss).astype(np.float32), sample_rate=rate)


def _peak(audio: AudioSegment) -> float:
    return float(np.max(np.abs(audio.data)))


class TestBoostQuietSpeech:
    def test_brings_speech_from_across_the_room_up_to_a_normal_level(self):
        boosted = boost_quiet_speech(_speech(0.1))

        assert _peak(boosted) == pytest.approx(0.5, rel=1e-3)

    def test_never_boosts_more_than_ten_times(self):
        boosted = boost_quiet_speech(_speech(0.02))

        assert _peak(boosted) == pytest.approx(0.2, rel=1e-3)

    def test_never_pushes_any_sample_past_full_scale(self):
        audio = _speech(0.05, noise=0.01)

        boosted = boost_quiet_speech(audio)

        assert _peak(boosted) <= 0.5 + 1e-6

    def test_leaves_speech_that_is_already_loud_enough_alone(self):
        audio = _speech(0.6)

        assert boost_quiet_speech(audio) is audio

    def test_leaves_a_near_silent_recording_alone_rather_than_raising_the_hiss(self):
        audio = _speech(0.004, noise=0.001)

        assert boost_quiet_speech(audio) is audio

    def test_skips_a_boost_too_small_to_matter(self):
        audio = _speech(0.49)

        assert boost_quiet_speech(audio) is audio

    def test_keeps_the_length_rate_and_float32_samples(self):
        audio = _speech(0.1, seconds=600.0)

        boosted = boost_quiet_speech(audio)

        assert boosted.sample_rate == 16000
        assert boosted.data.dtype == np.float32
        assert boosted.duration_seconds == pytest.approx(600.0)

    def test_does_not_change_the_recording_it_was_given(self):
        audio = _speech(0.1)
        before = audio.data.copy()

        boost_quiet_speech(audio)

        np.testing.assert_array_equal(audio.data, before)

    def test_a_single_loud_click_limits_the_boost_instead_of_clipping(self):
        data = _speech(0.05).data.copy()
        data[1000] = 0.9

        audio = AudioSegment(data=data, sample_rate=16000)

        assert boost_quiet_speech(audio) is audio
