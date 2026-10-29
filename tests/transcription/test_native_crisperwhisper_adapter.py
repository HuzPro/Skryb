"""The native CrisperWhisper engine's Python side, on fakes: no GPU, no DLL, no real weights."""

import ctypes
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.native_crisperwhisper_adapter import (
    NativeCrisperWhisperAdapter,
    _engine_config,
    _suppressed_tokens,
    _tensor_offsets,
)

_LARGE_CONFIG = {
    "d_model": 1280, "encoder_attention_heads": 20, "decoder_attention_heads": 20,
    "encoder_ffn_dim": 5120, "decoder_ffn_dim": 5120,
    "max_source_positions": 1500, "max_target_positions": 448,
    "num_mel_bins": 80, "encoder_layers": 32, "decoder_layers": 32, "eos_token_id": 50257,
}


def _write_safetensors(path: Path, tensors: list[tuple[str, str, list[int]]]) -> dict[str, int]:
    """A safetensors file of zeros; returns each tensor's offset in the file."""
    header, cursor = {}, 0
    for name, dtype, shape in tensors:
        size = int(np.prod(shape)) * (4 if dtype == "F32" else 2)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [cursor, cursor + size]}
        cursor += size
    header["__metadata__"] = {"format": "pt"}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(cursor))
    return {name: 8 + len(raw) + info["data_offsets"][0] for name, info in header.items() if name != "__metadata__"}


def _whisper_tensors(layers: int, vocab: int = 51896) -> list[tuple[str, str, list[int]]]:
    # The real checkpoint opens with a 2-byte bias, which shifts every later tensor off 16-byte alignment.
    tensors = [("encoder_blank_head.bias", "BF16", [1]), ("encoder_blank_head.weight", "BF16", [1, 1280])]
    tensors.append(("model.decoder.embed_tokens.weight", "BF16", [vocab, 8]))
    for layer in range(layers):
        for part in ("q_proj", "k_proj", "v_proj", "out_proj"):
            tensors.append((f"model.encoder.layers.{layer}.self_attn.{part}.weight", "BF16", [8, 8]))
    return tensors


class TestTensorOffsets:
    def test_each_tensor_is_found_at_its_first_byte_in_the_file(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.safetensors"
        expected = _write_safetensors(weights, _whisper_tensors(layers=2))

        names, offsets, vocab = _tensor_offsets(weights)

        assert dict(zip(names, offsets)) == {n: o for n, o in expected.items() if n.startswith("model.")}
        assert vocab == 51896

    def test_a_large_checkpoint_maps_every_one_of_its_tensors(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.safetensors"
        expected = _write_safetensors(weights, _whisper_tensors(layers=320, vocab=51897))

        names, offsets, vocab = _tensor_offsets(weights)

        assert len(names) == 1 + 320 * 4
        assert [expected[n] for n in names] == offsets
        assert vocab == 51897

    def test_the_extra_heads_outside_the_whisper_model_are_left_out(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.safetensors"
        _write_safetensors(weights, _whisper_tensors(layers=1))

        names, _, _ = _tensor_offsets(weights)

        assert not [name for name in names if name.startswith("encoder_blank_head")]

    def test_weights_not_in_bf16_are_refused_by_name(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.safetensors"
        _write_safetensors(weights, [("model.decoder.embed_tokens.weight", "F16", [4, 8])])

        with pytest.raises(TranscriptionError, match="embed_tokens.weight is F16"):
            _tensor_offsets(weights)

    def test_weights_at_mixed_alignments_are_refused(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.safetensors"
        _write_safetensors(weights, [
            ("model.decoder.embed_tokens.weight", "BF16", [4, 8]),
            ("model.decoder.layer_norm.bias", "BF16", [1]),
            ("model.decoder.layer_norm.weight", "BF16", [8]),
        ])

        with pytest.raises(TranscriptionError, match="aren't laid out the same way"):
            _tensor_offsets(weights)

    def test_a_checkpoint_without_a_token_embedding_is_refused(self, tmp_path: Path) -> None:
        weights = tmp_path / "model.safetensors"
        _write_safetensors(weights, [("model.decoder.layer_norm.weight", "BF16", [8])])

        with pytest.raises(TranscriptionError, match="no token embedding"):
            _tensor_offsets(weights)

    def test_a_missing_download_says_to_download_again(self, tmp_path: Path) -> None:
        with pytest.raises(TranscriptionError, match="download the model again"):
            _tensor_offsets(tmp_path / "model.safetensors")


class TestModelConfig:
    def test_the_engine_gets_the_models_own_shape(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(json.dumps(_LARGE_CONFIG), encoding="utf-8")

        config = _engine_config(tmp_path, vocab=51896)

        assert (config.n_mels, config.encoder_layers, config.decoder_layers, config.vocab, config.eos) == (
            80, 32, 32, 51896, 50257,
        )

    @pytest.mark.parametrize(("key", "value"), [("d_model", 1024), ("decoder_attention_heads", 16), ("max_target_positions", 512)])
    def test_a_whisper_of_another_size_is_refused_before_it_reaches_the_gpu(
        self, tmp_path: Path, key: str, value: int,
    ) -> None:
        (tmp_path / "config.json").write_text(json.dumps({**_LARGE_CONFIG, key: value}), encoding="utf-8")

        with pytest.raises(TranscriptionError, match=f"{key} is {value}"):
            _engine_config(tmp_path, vocab=51896)

    def test_suppressed_tokens_come_from_the_generation_config_without_its_negative_ids(self, tmp_path: Path) -> None:
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"suppress_tokens": [-1, 1, 2, 7, 50362, 1, 50363]}), encoding="utf-8",
        )

        assert _suppressed_tokens(tmp_path) == [1, 2, 7, 50362, 1, 50363]

    @pytest.mark.parametrize("suppress", [None, []])
    def test_a_generation_config_without_suppressed_tokens_suppresses_nothing(self, tmp_path: Path, suppress) -> None:
        (tmp_path / "generation_config.json").write_text(json.dumps({"suppress_tokens": suppress}), encoding="utf-8")

        assert _suppressed_tokens(tmp_path) == []


class _FakeTokens:
    def decoder_prompt(self, language: str, mode: str, context: str = "") -> list[int]:
        return [50358 + len(context), 50258, 50259, 50360, 50364] if mode == "verbatim" else [1, 2]

    def decode(self, token_ids) -> str:
        return " ".join(f"<{token}>" for token in token_ids)


class _FakeEngineLibrary:
    """Records what the adapter hands cw.dll and answers with fixed ids, or with an error."""

    def __init__(self, ids: list[int], error: str = "") -> None:
        self.ids, self.error, self.calls = ids, error, []

    def cw_transcribe(self, handle, samples, count, prompt, prompt_len, ids, max_ids, error, error_len):
        self.calls.append({"samples": count, "prompt": list(prompt[:prompt_len]), "max_ids": max_ids})
        if self.error:
            error.value = self.error.encode()
            return -1
        for index, token in enumerate(self.ids):
            ids[index] = token
        return len(self.ids)

    def cw_stats(self, handle) -> bytes:
        return b"{}"

    def cw_free(self, handle) -> None:
        self.freed = handle


def _loaded_adapter(tmp_path: Path, library: _FakeEngineLibrary, mode: str = "verbatim") -> NativeCrisperWhisperAdapter:
    adapter = NativeCrisperWhisperAdapter(tmp_path, "en", mode, dll_path=tmp_path / "cw.dll")
    adapter._lib, adapter._handle, adapter._tokens = library, 1, _FakeTokens()
    return adapter


class TestTranscribing:
    def test_the_prompt_and_the_audio_reach_the_engine_and_its_ids_are_decoded(self, tmp_path: Path) -> None:
        library = _FakeEngineLibrary(ids=[1820, 3157, 13])
        adapter = _loaded_adapter(tmp_path, library)

        result = adapter.transcribe(AudioSegment(np.full(16000 * 3, 0.1, dtype=np.float32), 16000))

        assert result.text == "<1820> <3157> <13>"
        assert library.calls == [{"samples": 48000, "prompt": [50358, 50258, 50259, 50360, 50364], "max_ids": 448}]

    def test_silence_that_decodes_to_nothing_gives_an_empty_transcript(self, tmp_path: Path) -> None:
        adapter = _loaded_adapter(tmp_path, _FakeEngineLibrary(ids=[]))

        assert adapter.transcribe(AudioSegment(np.zeros(16000, dtype=np.float32), 16000)).text == ""

    def test_an_engine_failure_surfaces_with_the_engines_own_message(self, tmp_path: Path) -> None:
        adapter = _loaded_adapter(tmp_path, _FakeEngineLibrary(ids=[], error="hipMalloc: out of memory"))

        with pytest.raises(TranscriptionError, match="hipMalloc: out of memory"):
            adapter.transcribe(AudioSegment(np.zeros(16000, dtype=np.float32), 16000))

    def test_audio_at_another_sample_rate_is_refused(self, tmp_path: Path) -> None:
        adapter = _loaded_adapter(tmp_path, _FakeEngineLibrary(ids=[]))

        with pytest.raises(TranscriptionError, match="16000 Hz"):
            adapter.transcribe(AudioSegment(np.zeros(44100, dtype=np.float32), 44100))

    def test_transcribing_before_loading_is_refused(self, tmp_path: Path) -> None:
        adapter = NativeCrisperWhisperAdapter(tmp_path, "en")

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(AudioSegment(np.zeros(16000, dtype=np.float32), 16000))

    def test_disposing_frees_the_engine_once(self, tmp_path: Path) -> None:
        library = _FakeEngineLibrary(ids=[])
        adapter = _loaded_adapter(tmp_path, library)

        adapter.dispose()
        adapter.dispose()

        assert library.freed == 1
        assert adapter.is_model_loaded() is False


def test_loading_without_the_engine_dll_says_how_to_get_it(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    _write_safetensors(weights, _whisper_tensors(layers=1))
    (tmp_path / "config.json").write_text(json.dumps(_LARGE_CONFIG), encoding="utf-8")
    (tmp_path / "generation_config.json").write_text(json.dumps({"suppress_tokens": [1]}), encoding="utf-8")
    adapter = NativeCrisperWhisperAdapter(tmp_path, "en", dll_path=tmp_path / "cw.dll")
    adapter_module = sys.modules[NativeCrisperWhisperAdapter.__module__]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(adapter_module, "WhisperTokens", lambda model_dir: _FakeTokens())
        with pytest.raises(TranscriptionError, match="native/whisper/build.sh"):
            adapter.load_model()
    assert adapter.is_model_loaded() is False


def test_the_engines_config_struct_matches_the_dll_header() -> None:
    from skryb.transcription.native_crisperwhisper_adapter import _CwConfig

    assert ctypes.sizeof(_CwConfig) == 5 * ctypes.sizeof(ctypes.c_int)


def test_the_engine_never_imports_torch_or_transformers() -> None:
    probe = (
        "import sys, skryb.transcription.native_crisperwhisper_adapter;"
        "print(any(m in sys.modules for m in ('torch', 'transformers')))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "False"
