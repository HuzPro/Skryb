"""The torch-free log-mel front end, pinned against the Transformers one."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.whisper_features import (
    log_mel_spectrogram,
    transcribe_in_windows,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SAMPLE_RATE = 16000
_ENCODER_WINDOW_SAMPLES = 30 * _SAMPLE_RATE
_MEL_BIN_COUNT = 128
_ENCODER_FRAMES = 3000
# ``WhisperFeatureExtractor`` runs its STFT through torch when torch is
# installed and through numpy otherwise, and those two paths differ from
# each other by ~2e-5 on real speech. Anything under this tolerance is
# inside Transformers' own spread, not a discrepancy of ours.
_REFERENCE_TOLERANCE = 1e-4


def _speech(seconds: float) -> AudioSegment:
    """A chirp plus noise: broadband, so every mel bin carries energy."""
    generator = np.random.default_rng(20260907)
    sample_count = int(seconds * _SAMPLE_RATE)
    elapsed = np.arange(sample_count, dtype=np.float64) / _SAMPLE_RATE
    sweep = np.sin(2 * np.pi * (200.0 + 300.0 * elapsed) * elapsed)
    noise = generator.normal(0.0, 0.05, size=sample_count)
    data = (0.4 * sweep + noise).astype(np.float32)
    return AudioSegment(data=data, sample_rate=_SAMPLE_RATE)


class TestLogMelSpectrogram:
    def test_should_produce_the_encoder_input_shape(self) -> None:
        features = log_mel_spectrogram(_speech(1.5))

        assert features.shape == (_MEL_BIN_COUNT, _ENCODER_FRAMES)
        assert features.dtype == np.float32

    def test_should_reject_audio_recorded_at_another_rate(self) -> None:
        audio = AudioSegment(
            data=np.zeros(8000, dtype=np.float32),
            sample_rate=8000,
        )

        with pytest.raises(ValueError, match="16000 Hz"):
            log_mel_spectrogram(audio)

    def test_should_ignore_audio_past_the_encoder_window(self) -> None:
        long_audio = _speech(31.0)
        clipped = AudioSegment(
            data=long_audio.data[:_ENCODER_WINDOW_SAMPLES],
            sample_rate=_SAMPLE_RATE,
        )

        assert np.array_equal(
            log_mel_spectrogram(long_audio),
            log_mel_spectrogram(clipped),
        )

    def test_should_clamp_to_eight_decades_below_the_peak(self) -> None:
        # The 28.5 s of zero padding sits on the floor, so the full
        # eight-decade span survives Whisper's ``(x + 4) / 4`` rescale.
        features = log_mel_spectrogram(_speech(1.5))

        assert features.max() - features.min() == pytest.approx(2.0)

    @pytest.mark.integration
    @pytest.mark.parametrize("seconds", [1.5, 30.0])
    def test_should_match_the_transformers_extractor(
        self,
        seconds: float,
    ) -> None:
        transformers = pytest.importorskip("transformers")
        audio = _speech(seconds)
        reference = transformers.WhisperFeatureExtractor(
            feature_size=_MEL_BIN_COUNT,
        )(
            audio.data,
            sampling_rate=_SAMPLE_RATE,
            return_tensors="np",
        ).input_features[0]

        features = log_mel_spectrogram(audio)

        assert np.abs(features - reference).max() < _REFERENCE_TOLERANCE


def _speech_then_silence(speech_seconds: float, total_seconds: float) -> AudioSegment:
    speech = _speech(speech_seconds).data
    data = np.zeros(int(total_seconds * _SAMPLE_RATE), dtype=np.float32)
    data[:speech.size] = speech
    return AudioSegment(data=data, sample_rate=_SAMPLE_RATE)


class _ScriptedWindows:
    """Stands in for the model: one scripted transcript per window."""

    def __init__(self, *transcripts: str) -> None:
        self._transcripts = list(transcripts)
        self.contexts: list[str] = []

    def __call__(self, window: AudioSegment, context: str) -> str:
        self.contexts.append(context)
        return self._transcripts.pop(0)


class TestTranscribeInWindows:
    def test_should_transcribe_short_audio_in_one_window(self) -> None:
        windows = _ScriptedWindows("Hello there.")

        text = transcribe_in_windows(_speech(12.0), windows)

        assert text == "Hello there."
        assert windows.contexts == [""]

    def test_should_keep_a_sentence_whole_across_the_window_seam(self) -> None:
        windows = _ScriptedWindows(
            "You would have to wait until at least the middle of next week.",
            "next week before going anywhere at all.",
        )

        text = transcribe_in_windows(_speech(50.0), windows)

        assert text == (
            "You would have to wait until at least the middle of next week "
            "before going anywhere at all."
        )

    def test_should_prompt_each_window_with_the_last_confirmed_words(
        self,
    ) -> None:
        first = " ".join(f"w{index}" for index in range(1, 21))
        windows = _ScriptedWindows(first, "rest.")

        transcribe_in_windows(_speech(50.0), windows)

        assert windows.contexts[1] == " ".join(
            f"w{index}" for index in range(7, 19)
        )

    def test_should_keep_every_word_when_the_window_ends_in_silence(
        self,
    ) -> None:
        windows = _ScriptedWindows("That is all for today.", "")

        text = transcribe_in_windows(_speech_then_silence(20.0, 40.0), windows)

        assert text == "That is all for today."
        assert windows.contexts[1] == "That is all for today."

    def test_should_keep_seam_words_spoken_before_the_next_window(
        self,
    ) -> None:
        windows = _ScriptedWindows(
            "They read it on Monday morning.",
            "And after that we plan.",
        )

        text = transcribe_in_windows(_speech_then_silence(27.0, 50.0), windows)

        assert text == "They read it on Monday morning. And after that we plan."

    def test_should_let_the_next_window_redo_the_words_the_seam_cut(
        self,
    ) -> None:
        windows = _ScriptedWindows(
            "After that we can finally see.",
            "finally schedule the meeting.",
        )

        text = transcribe_in_windows(_speech(50.0), windows)

        assert text == "After that we can finally schedule the meeting."

    def test_should_keep_the_last_words_of_the_final_window(self) -> None:
        windows = _ScriptedWindows("one two three four", "three four five six.")

        text = transcribe_in_windows(_speech(40.0), windows)

        assert text == "one two three four five six."


class TestFeatureExtractionDependencies:
    def test_should_import_without_torch_or_transformers(self) -> None:
        """The whole point: the CT2 engine must not pay for either import."""
        probe = (
            "import sys;"
            "import voicetotext.transcription.whisper_features;"
            "print('torch' in sys.modules, 'transformers' in sys.modules)"
        )

        completed = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )

        assert completed.stdout.strip() == "False False"
