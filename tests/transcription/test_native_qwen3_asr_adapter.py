"""The native Qwen3-ASR adapter and its packer, without the DLL, a GPU or real weights.

Real-weights accuracy and speed are checked by hand with native/bench
(reference.py, qasr-bench.exe, compare.py) against the PyTorch reference.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription import qwen3_asr_pack
from voicetotext.transcription.native_qwen3_asr_adapter import (
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
