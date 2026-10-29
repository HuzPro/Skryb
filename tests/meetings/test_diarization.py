"""The streaming diarizer's NumPy side: chunking, features and the speaker cache. No GPU, no torch, no weights."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from skryb.meetings.diarization import (
    DiarizerStream,
    StreamingFeatures,
    slaney_mel_filters,
)
from skryb.meetings.segmenter import SILENCE

# The checkpoint's own values (nvidia/Nemotron-3-Diarization config.json and processor_config.json).
_FEATURES = {"feature_size": 128, "hop_length": 160, "n_fft": 512, "preemphasis": 0.97, "sampling_rate": 16000,
             "win_length": 400}
_STREAMING = {"fifo_length": 264, "speaker_cache_length": 264, "speaker_cache_update_period": 222,
              "speaker_cache_silence_frames_per_speaker": 1, "prediction_score_threshold": 0.25,
              "latest_frames_score_boost": 0.05, "min_positive_scores_rate": 0.5, "strong_boost_rate": 0.75,
              "weak_boost_rate": 1.5, "num_speakers": 8, "subsampling_factor": 8}
_FIRST_CHUNK_SAMPLES = 103 * 160 + 200  # 104 mel frames with centred windows
_LATER_CHUNK_SAMPLES = 104 * 160 + 400  # 104 mel frames without


def _snapshot(tmp_path: Path) -> Path:
    config = {"audio_config": {"subsampling_factor": 8, "num_hidden_layers": 31}, "streaming_config": _STREAMING}
    processor = {"feature_extractor": _FEATURES, "streaming_mode": "low_latency",
                 "streaming_modes": {"low_latency": [9, 4], "very_low_latency": [6, 2]}}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "processor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    return tmp_path


def _speech(seconds: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16000)) / 16000
    return (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


class TestMelFilters:
    def test_they_match_librosas_slaney_filters(self) -> None:
        filters = slaney_mel_filters(16000, 512, 128)

        assert filters.shape == (128, 257) and filters.dtype == np.float32
        assert filters[0, 1] == pytest.approx(0.02837754227221012, rel=1e-6)
        assert filters[127, 250] == pytest.approx(0.005223188549280167, rel=1e-6)
        assert filters[64, 100] == 0.0
        assert int(filters[64].argmax()) == 55
        assert float(filters.sum()) == pytest.approx(4.090487480163574, rel=1e-6)


class TestStreamingFeatures:
    @pytest.fixture
    def features(self) -> StreamingFeatures:
        return StreamingFeatures(_FEATURES)

    def test_a_first_chunk_of_the_processors_size_gives_its_104_frames(self, features: StreamingFeatures) -> None:
        assert features(_speech(1.1)[:_FIRST_CHUNK_SAMPLES], center=True, last=False).shape == (104, 128)

    def test_a_later_chunk_of_the_processors_size_gives_its_104_frames(self, features: StreamingFeatures) -> None:
        assert features(_speech(1.1)[:_LATER_CHUNK_SAMPLES], center=False, last=False).shape == (104, 128)

    def test_the_last_chunk_is_padded_so_it_keeps_its_final_frame(self, features: StreamingFeatures) -> None:
        tail = _speech(0.4)[:3000]

        assert len(features(tail, center=False, last=True)) == len(features(tail, center=False, last=False)) + 1

    def test_silence_comes_out_at_the_log_floor(self, features: StreamingFeatures) -> None:
        out = features(np.zeros(_LATER_CHUNK_SAMPLES, np.float32), center=False, last=False)

        assert np.allclose(out, np.log(np.float32(2.0 ** -24)))

    def test_a_chunk_too_short_for_one_window_has_no_frames(self, features: StreamingFeatures) -> None:
        assert features(np.zeros(100, np.float32), center=False, last=False).shape == (0, 128)

    def test_a_streamed_chunk_matches_the_same_frames_of_one_centred_pass(self, features: StreamingFeatures) -> None:
        audio = _speech(3.0)
        whole = features(audio, center=True, last=False)
        start = 72 * 160 - 256  # the processor's audio_chunk_start for the second chunk

        later = features(audio[start:start + _LATER_CHUNK_SAMPLES], center=False, last=False)

        # Pre-emphasis restarts at the chunk's first sample, so only its very first frame may differ.
        assert np.allclose(later[1:], whole[73:73 + 103], atol=1e-4)


class _RecordingStep:
    def __init__(self, speaker: int) -> None:
        self.calls: list[tuple[int, int]] = []
        self.speaker = speaker

    def __call__(self, features: np.ndarray, lookahead: int) -> np.ndarray:
        self.calls.append((len(features), lookahead))
        frames = len(features) if lookahead == 0 else (len(features) // 8 - lookahead) * 8
        logits = np.full((frames, 8), -8.0, np.float32)
        logits[: frames // 2, self.speaker] = 8.0  # speech in the first half of every chunk
        return logits


class TestDiarizerStream:
    def test_a_long_meeting_fed_in_small_pieces_is_cut_the_way_the_processor_cuts_it(self, tmp_path: Path) -> None:
        step = _RecordingStep(speaker=2)
        stream = DiarizerStream(_snapshot(tmp_path), step)
        audio = _speech(60.0)

        labels = []
        for start in range(0, len(audio), 1234):
            labels += stream.push(audio[start:start + 1234], last=start + 1234 >= len(audio))

        assert step.calls[0] == (104, 4)
        assert set(step.calls[1:-1]) == {(104, 4)}
        assert step.calls[-1][1] == 0
        assert len(labels) == pytest.approx(6000, abs=8)  # one label per 10 ms
        assert set(labels) == {2, SILENCE}

    def test_audio_already_scored_is_let_go_of(self, tmp_path: Path) -> None:
        stream = DiarizerStream(_snapshot(tmp_path), _RecordingStep(speaker=0))

        for start in range(0, 16000 * 120, 8000):
            stream.push(_speech(0.5, seed=start))

        assert len(stream._audio) < 2 * _LATER_CHUNK_SAMPLES

    def test_a_meeting_shorter_than_one_window_gives_no_labels(self, tmp_path: Path) -> None:
        step = _RecordingStep(speaker=0)

        assert DiarizerStream(_snapshot(tmp_path), step).push(np.zeros(300, np.float32), last=True) == []
        assert step.calls == []










def test_the_diarization_module_never_imports_torch() -> None:
    probe = "import sys, skryb.meetings.diarization; print('torch' in sys.modules or 'transformers' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False"
