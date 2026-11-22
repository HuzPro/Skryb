"""The CPU diarizer's bridge to NeMo-Speech.cpp, against a stand-in for NVIDIA's library: no DLL, no model."""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from skryb.core.exceptions import TranscriptionError
from skryb.meetings import cpu_diarizer
from skryb.meetings.cpu_diarizer import CpuDiarizer
from skryb.meetings.segmenter import SILENCE

_SPEAKERS = 8
_SAMPLES_PER_FRAME = 160  # 10 ms at 16 kHz


class FakeNemoSpeech:
    """NeMo-Speech.cpp's diarization C ABI over a scripted [frames][speakers] probability table.

    Like the real one it labels whole chunks only, keeps the last ``horizon`` frames' probabilities (older ones are
    folded into segments) and labels the tail on finish.
    """

    def __init__(self, probs: np.ndarray, *, chunk: int = 100, horizon: int = 10**9) -> None:
        self.table = np.ascontiguousarray(probs, np.float32)
        self.chunk, self.horizon = chunk, horizon
        self.samples = 0
        self.finished = False
        self.model_path: bytes | None = None
        self.fail_on: str | None = None
        self.destroyed = self.closed = 0
        self.pushed_rates: set[int] = set()

    # ---- the C ABI ----
    def nemo_speech_diar_create(self, config, out) -> int:
        self.model_path = config._obj.model_path
        assert config._obj.gpu == -1 and config._obj.size == ctypes.sizeof(cpu_diarizer._ModelConfig)
        assert config._obj.left_context_frames == -1, "0 is an explicit left context, not 'keep the preset'"
        out._obj.value = 1
        return self._status("create")

    def nemo_speech_diar_stream_open(self, model, out) -> int:
        out._obj.value = 2
        return self._status("open")

    def nemo_speech_diar_num_speakers(self, model) -> int:
        return _SPEAKERS

    def nemo_speech_diar_seconds_per_frame(self, model) -> float:
        return 0.01

    def nemo_speech_diar_stream_push_f32(self, stream, samples, count: int, rate: int) -> int:
        self.samples += count
        self.pushed_rates.add(rate)
        return self._status("push")

    def nemo_speech_diar_stream_finish(self, stream) -> int:
        self.finished = True
        return self._status("finish")

    def nemo_speech_diar_frame_count(self, stream) -> int:
        frames = min(self.samples // _SAMPLES_PER_FRAME, len(self.table))
        return frames if self.finished else frames // self.chunk * self.chunk

    def nemo_speech_diar_frame_probs_start(self, stream) -> int:
        return max(0, self.nemo_speech_diar_frame_count(stream) - self.horizon)

    def nemo_speech_diar_frame_probs(self, stream, out, capacity: int) -> int:
        start, end = self.nemo_speech_diar_frame_probs_start(stream), self.nemo_speech_diar_frame_count(stream)
        retained = self.table[start:end].ravel()
        if capacity < retained.size:
            return 1
        ctypes.memmove(out, retained.ctypes.data, retained.nbytes)
        return self._status("read")

    def nemo_speech_diar_stream_close(self, stream) -> None:
        self.closed += 1

    def nemo_speech_diar_destroy(self, model) -> None:
        self.destroyed += 1

    def nemo_speech_asr_last_error(self) -> bytes:
        return "ggml_cpu: failed to allocate 512.00 MiB (Speicher voll, ünïcödé)".encode()

    def _status(self, step: str) -> int:
        return 3 if self.fail_on == step else 0


def _diarizer(monkeypatch: pytest.MonkeyPatch, lib: FakeNemoSpeech, model: Path = Path("model.gguf")) -> CpuDiarizer:
    monkeypatch.setattr(cpu_diarizer, "_open_library", lambda path: lib)
    return CpuDiarizer(model, Path("nemo_speech_asr_c.dll"))


def _one_hot(speakers: list[int], level: float = 0.9) -> np.ndarray:
    probs = np.full((len(speakers), _SPEAKERS), 0.05, np.float32)
    for frame, speaker in enumerate(speakers):
        if speaker != SILENCE:
            probs[frame, speaker] = level
    return probs


def _feed(diarizer: CpuDiarizer, frames: int, push_samples: int = 1600) -> list[int]:
    labels: list[int] = []
    audio = np.zeros(frames * _SAMPLES_PER_FRAME, np.float32)
    for i in range(0, len(audio), push_samples):
        labels += diarizer.push(audio[i:i + push_samples])
    return labels + diarizer.push(np.zeros(0, np.float32), last=True)


def test_each_frame_gets_its_strongest_speaker_and_a_frame_at_half_probability_or_below_is_silence(monkeypatch) -> None:
    probs = np.array([
        [0.9, 0.1, 0, 0, 0, 0, 0, 0],  # speaker 0
        [0.2, 0.0, 0, 0, 0, 0, 0.7, 0],  # speaker 6
        [0.5, 0.5, 0, 0, 0, 0, 0, 0],  # exactly 0.5 is not speech
        [0.0, 0.0, 0, 0, 0, 0, 0, 0.500001],  # just above is
        [0.8, 0.8, 0, 0, 0, 0, 0, 0],  # a tie goes to the speaker who arrived first
        [0.0] * 8,
    ], np.float32)
    diarizer = _diarizer(monkeypatch, FakeNemoSpeech(probs, chunk=1))

    assert _feed(diarizer, len(probs)) == [0, 6, SILENCE, 7, 0, SILENCE]


def test_ten_minutes_pushed_a_tenth_of_a_second_at_a_time_get_one_label_per_frame_in_order(monkeypatch) -> None:
    rng = np.random.default_rng(7)
    script = rng.choice([SILENCE, 0, 1, 2, 3, 4, 5, 6, 7], size=60_000).tolist()
    diarizer = _diarizer(monkeypatch, FakeNemoSpeech(_one_hot(script), chunk=48))

    assert _feed(diarizer, len(script)) == script


def test_labels_come_out_a_chunk_at_a_time_while_the_meeting_runs_not_only_at_the_end(monkeypatch) -> None:
    diarizer = _diarizer(monkeypatch, FakeNemoSpeech(_one_hot([1] * 1000), chunk=100))

    first = diarizer.push(np.zeros(150 * _SAMPLES_PER_FRAME, np.float32))
    second = diarizer.push(np.zeros(0, np.float32))

    assert first == [1] * 100
    assert second == []


def test_the_last_partial_chunk_is_labelled_when_the_meeting_ends(monkeypatch) -> None:
    diarizer = _diarizer(monkeypatch, FakeNemoSpeech(_one_hot([2] * 250), chunk=100))

    during = diarizer.push(np.zeros(250 * _SAMPLES_PER_FRAME, np.float32))
    at_the_end = diarizer.push(np.zeros(0, np.float32), last=True)

    assert (len(during), at_the_end) == (200, [2] * 50)


def test_frames_the_library_folded_into_segments_are_neither_read_twice_nor_skipped(monkeypatch) -> None:
    script = ([0] * 700 + [1] * 700 + [SILENCE] * 300) * 3
    diarizer = _diarizer(monkeypatch, FakeNemoSpeech(_one_hot(script), chunk=100, horizon=250))

    assert _feed(diarizer, len(script)) == script


def test_frames_folded_away_before_they_were_read_still_count_as_silence_so_the_timeline_stays_whole(monkeypatch) -> None:
    lib = FakeNemoSpeech(_one_hot([3] * 1000), chunk=100, horizon=300)
    diarizer = _diarizer(monkeypatch, lib)

    labels = diarizer.push(np.zeros(1000 * _SAMPLES_PER_FRAME, np.float32))

    assert labels == [SILENCE] * 700 + [3] * 300


def test_audio_goes_in_at_sixteen_kilohertz_and_empty_pushes_change_nothing(monkeypatch) -> None:
    lib = FakeNemoSpeech(_one_hot([0] * 300), chunk=100)
    diarizer = _diarizer(monkeypatch, lib)

    assert diarizer.push(np.zeros(0, np.float32)) == []
    diarizer.push(np.zeros(1600, np.float64))

    assert lib.samples == 1600 and lib.pushed_rates == {16000}
    assert diarizer.frame_seconds == 0.01


def test_a_model_path_with_accents_and_chinese_reaches_the_library_as_utf8(monkeypatch) -> None:
    lib = FakeNemoSpeech(_one_hot([0]))
    model = Path("C:/Users/Zoë Ñúñez/AppData/Roaming/Skryb/models/模型/Nemotron-3-Diarization.q8_0.gguf")

    _diarizer(monkeypatch, lib, model)

    assert lib.model_path == str(model).encode("utf-8")


@pytest.mark.parametrize("step", ["create", "open", "push", "finish", "read"])
def test_a_failing_library_call_raises_with_the_librarys_own_reason(monkeypatch, step: str) -> None:
    lib = FakeNemoSpeech(_one_hot([0] * 200), chunk=100)
    lib.fail_on = step

    with pytest.raises(TranscriptionError, match=r"status 3.*failed to allocate 512\.00 MiB \(Speicher voll, ünïcödé\)"):
        _feed(_diarizer(monkeypatch, lib), 200)


def test_letting_go_frees_the_stream_and_the_model_once_even_when_asked_twice(monkeypatch) -> None:
    lib = FakeNemoSpeech(_one_hot([0]))
    diarizer = _diarizer(monkeypatch, lib)

    diarizer.dispose()
    diarizer.dispose()

    assert (lib.closed, lib.destroyed) == (1, 1)


def test_without_nvidias_library_on_disk_it_says_the_cpu_diarizer_isnt_installed(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError, match="isn't installed"):
        CpuDiarizer(tmp_path / "model.gguf", tmp_path / "nemo-speech" / "nemo_speech_asr_c.dll")


def test_a_library_file_that_is_not_a_loadable_dll_fails_with_a_clear_error(tmp_path: Path) -> None:
    broken = tmp_path / "nemo_speech_asr_c.dll"
    broken.write_bytes(b"MZ" + b"\0" * 64)

    with pytest.raises(TranscriptionError, match="didn't load"):
        CpuDiarizer(tmp_path / "model.gguf", broken)
