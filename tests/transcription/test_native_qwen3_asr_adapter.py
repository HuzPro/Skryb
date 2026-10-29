"""The native Qwen3-ASR adapter and its packer, without the DLL, a GPU or real weights.

Real-weights accuracy and speed are checked by hand with native/bench
(reference.py, qasr-bench.exe, compare.py) against the PyTorch reference.
"""

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription import qwen3_asr_pack
from skryb.transcription.native_qwen3_asr_adapter import (
    NativeQwen3AsrAdapter,
    packed_model_path,
)


def test_the_packed_file_lives_inside_the_snapshot_it_came_from(tmp_path: Path) -> None:
    assert packed_model_path(tmp_path) == tmp_path / ".qasr" / "qwen3-asr.bin"


def test_a_missing_dll_fails_with_the_build_command(tmp_path: Path) -> None:
    packed_model_path(tmp_path).parent.mkdir()
    packed_model_path(tmp_path).write_bytes(b"QASR0001")
    adapter = NativeQwen3AsrAdapter(tmp_path, dll_path=tmp_path / "missing.dll")

    with pytest.raises(TranscriptionError, match="build.sh"):
        adapter.load_model()


def test_transcribing_before_loading_is_refused(tmp_path: Path) -> None:
    adapter = NativeQwen3AsrAdapter(tmp_path, dll_path=tmp_path / "missing.dll")
    audio = AudioSegment(data=np.zeros(16000, dtype=np.float32), sample_rate=16000)

    with pytest.raises(TranscriptionError, match="not loaded"):
        adapter.transcribe(audio)


class _FakeQasr:
    """Stands in for qasr.dll: writes a fixed transcript into the caller's buffer."""

    def __init__(self, transcript: str, question_probability: float = -1.0) -> None:
        self._transcript = transcript.encode()
        self._question_probability = question_probability

    def qasr_question_probability(self, handle) -> float:  # noqa: ANN001
        return self._question_probability

    def qasr_transcribe(self, handle, samples, count, buffer, size) -> int:  # noqa: ANN001
        ctypes.memmove(buffer, self._transcript, len(self._transcript))
        return len(self._transcript)

    def qasr_stats(self, handle) -> bytes:  # noqa: ANN001
        return b"{}"


def test_spoken_urdu_written_in_hindi_script_is_typed_as_roman_urdu(tmp_path: Path) -> None:
    adapter = NativeQwen3AsrAdapter(tmp_path)
    adapter._lib, adapter._handle = _FakeQasr("Slack से निकाल रहे हैं, ठीक है?"), 1
    audio = AudioSegment(data=np.zeros(16000, dtype=np.float32), sample_rate=16000)

    assert adapter.transcribe(audio).text == "Slack se nikal rahe hain, thik hai?"


def test_dispose_before_loading_does_nothing(tmp_path: Path) -> None:
    NativeQwen3AsrAdapter(tmp_path).dispose()


def test_the_byte_decoder_maps_all_256_bytes_back_exactly_once() -> None:
    decoder = qwen3_asr_pack._byte_decoder()

    assert sorted(decoder.values()) == list(range(256))
    assert decoder["Ġ"] == ord(" ")
    assert decoder["Ċ"] == ord("\n")


def test_the_vocab_keeps_token_bytes_and_blanks_only_special_tokens(tmp_path: Path) -> None:
    tokenizer = {
        "model": {"vocab": {"Hello": 0, "Ġworld": 1, "Ã©": 2, "Ġ": 3}},
        "added_tokens": [
            {"id": 151645, "content": "<|im_end|>", "special": True},
            {"id": 151704, "content": "<asr_text>", "special": False},
        ],
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    offsets, blob = qwen3_asr_pack._vocab(tmp_path)

    def piece(i: int) -> bytes:
        return blob[offsets[i]:offsets[i + 1]].tobytes()

    assert len(offsets) == 151936 + 1
    assert piece(0) == b"Hello"
    assert piece(1) == b" world"
    assert piece(2) == "é".encode()
    assert piece(151645) == b""
    assert piece(151704) == b"<asr_text>"
    assert piece(5000) == b""


def _transcribe(tmp_path: Path, transcript: str, probability: float, threshold: float | None = 0.3) -> str:
    adapter = NativeQwen3AsrAdapter(tmp_path, question_threshold=threshold)
    adapter._lib, adapter._handle = _FakeQasr(transcript, probability), 1
    return adapter.transcribe(AudioSegment(data=np.zeros(16000, dtype=np.float32), sample_rate=16000)).text


class TestHearingQuestions:
    def test_a_full_stop_the_engine_nearly_made_a_question_mark_becomes_one(self, tmp_path: Path) -> None:
        assert _transcribe(tmp_path, "Why not store it all in the same storage.", 0.37) == (
            "Why not store it all in the same storage?"
        )

    def test_a_full_stop_the_engine_was_sure_of_stays(self, tmp_path: Path) -> None:
        assert _transcribe(tmp_path, "Do the prod push stuff.", 0.04) == "Do the prod push stuff."

    def test_the_bar_itself_counts_as_a_question(self, tmp_path: Path) -> None:
        assert _transcribe(tmp_path, "This one, right.", 0.3) == "This one, right?"

    def test_switched_off_it_leaves_even_a_near_certain_question_alone(self, tmp_path: Path) -> None:
        assert _transcribe(tmp_path, "How long.", 0.99, threshold=None) == "How long."

    def test_an_ellipsis_is_never_turned_into_a_question(self, tmp_path: Path) -> None:
        assert _transcribe(tmp_path, "So I was thinking...", 0.9) == "So I was thinking..."

    def test_a_question_mark_the_engine_wrote_itself_is_kept(self, tmp_path: Path) -> None:
        assert _transcribe(tmp_path, "How big is the damage?", -1.0) == "How big is the damage?"

    def test_only_the_last_full_stop_changes(self, tmp_path: Path) -> None:
        assert _transcribe(tmp_path, "Are you sure. Like we're good to disable the keys.", 0.5) == (
            "Are you sure. Like we're good to disable the keys?"
        )

    def test_urdu_in_hindi_script_still_becomes_roman_urdu_after_the_change(self, tmp_path: Path) -> None:
        assert _transcribe(tmp_path, "हाँ जी क्या बात हुई.", 0.4) == "haan ji kya baat hui?"

    def test_an_empty_transcript_stays_empty(self, tmp_path: Path) -> None:
        assert _transcribe(tmp_path, "", 0.9) == ""
