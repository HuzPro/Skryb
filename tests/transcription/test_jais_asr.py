"""Jais-ASR on the native engine: the 4-bit packing, the shape the engine reads, the prompt and the decision tag."""

import ctypes
from pathlib import Path

import numpy as np
import pytest

from skryb.recording.audio_segment import AudioSegment
from skryb.transcription import jais_asr_pack
from skryb.transcription.jais_asr_pack import engine_config, prompt_text, quantize_q4
from skryb.transcription.native_jais_asr_adapter import NativeJaisAsrAdapter, transcript_text, without_decision_tag
from skryb.transcription.qwen3_asr_prompt import PromptedTranscriber


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    """float32 values rounded to bf16 the way torch does, as raw bits."""
    u = values.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _slow_q4(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One group at a time, written straight from the engine's quantize_q4_kernel."""
    n, k = bits.shape
    packed = np.zeros((n, k // 2), np.uint8)
    scales = np.zeros((n, k // 32), np.float16)
    for r in range(n):
        for g in range(k // 32):
            x = (bits[r, g * 32:(g + 1) * 32].astype(np.uint32) << 16).view(np.float32)
            amax = np.float32(np.abs(x).max())
            sh = np.float16(amax / np.float32(7)) if amax > 0 else np.float16(1)
            if sh == 0:
                sh = np.float16(1)
            inv = np.float32(1) / np.float32(sh)
            for j in range(32):
                v = int(np.clip(np.rint(x[j] * inv), -8, 7)) + 8
                packed[r, g * 16 + j // 2] |= v << (4 * (j % 2))
            scales[r, g] = sh
    return packed, scales


def _realistic_weights(rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, (rows, cols)).astype(np.float32)
    w[0, :32] = 0  # a group of zeros
    w[1, :32] = 1e-9  # too small for an fp16 scale
    w[2, :32] = np.linspace(-7, 7, 32) * 0.01  # exact ties at half a step
    w[3, 32:64] = w[3, 0:32]  # a repeated group
    w[4, :] = -0.5  # a constant row
    return _bf16_bits(w)


def test_four_bit_packing_matches_the_engines_own_quantizer_group_for_group() -> None:
    bits = _realistic_weights(40, 128, seed=1)

    packed, scales = quantize_q4(bits)
    slow_packed, slow_scales = _slow_q4(bits)

    np.testing.assert_array_equal(scales.view(np.uint16), slow_scales.view(np.uint16))
    np.testing.assert_array_equal(packed, slow_packed)


def test_four_bit_packing_spans_row_blocks_without_a_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    bits = _realistic_weights(1000, 64, seed=2)
    whole_packed, whole_scales = quantize_q4(bits)

    monkeypatch.setattr(jais_asr_pack, "_ROW_BLOCK", 97)
    packed, scales = quantize_q4(bits)

    np.testing.assert_array_equal(packed, whole_packed)
    np.testing.assert_array_equal(scales.view(np.uint16), whole_scales.view(np.uint16))


def test_unpacked_weights_land_within_half_a_step_of_the_originals() -> None:
    bits = _realistic_weights(64, 256, seed=3)
    packed, scales = quantize_q4(bits)

    low, high = (packed & 15).astype(np.int8) - 8, (packed >> 4).astype(np.int8) - 8
    q = np.stack([low, high], axis=-1).reshape(64, 256)
    step = np.repeat(scales.astype(np.float32), 32, axis=1)
    original = (bits.astype(np.uint32) << 16).view(np.float32)

    assert np.all(np.abs(q * step - original) <= step / 2 + 1e-7)


def _jais_config() -> dict:
    return {
        "eos_token_id": 1,
        "audio_config": {"encoder_layers": 32, "d_model": 1280, "encoder_attention_heads": 20, "encoder_ffn_dim": 5120,
                         "output_dim": 2048},
        "text_config": {"num_hidden_layers": 36, "hidden_size": 2560, "num_attention_heads": 32, "num_key_value_heads": 8,
                        "intermediate_size": 10240, "vocab_size": 250624, "head_dim": 128, "rope_head_dim": 128,
                        "query_key_norm": False, "layernorm_num_groups": 2, "attention_gate_func": None,
                        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"}},
    }


def test_the_engine_is_told_jais_asrs_shape_in_the_order_it_reads_it() -> None:
    ints = engine_config(_jais_config())

    assert ints.dtype == np.int32 and ints.size == 18
    assert ints[:17].tolist() == [32, 1280, 20, 5120, 2048, 36, 2560, 32, 8, 10240, 250624, 1, 250019, 0, 2, 1, 1]
    assert ints[17:].view(np.float32)[0] == np.float32(1e7)


def test_a_checkpoint_with_a_gated_attention_or_partial_rope_is_refused() -> None:
    config = _jais_config()
    config["text_config"]["attention_gate_func"] = "silu"

    with pytest.raises(AssertionError):
        engine_config(config)


def test_code_switch_mode_is_the_system_turn_and_off_leaves_no_system_turn() -> None:
    on_prefix, suffix = prompt_text(code_switch=True)
    off_prefix, _ = prompt_text(code_switch=False)

    assert on_prefix == "<|ifm|begin_of_text|><|ifm|im_start|>system\n[Code-switch]<|ifm|im_end|><|ifm|im_start|>user\n<audio>"
    assert off_prefix == "<|ifm|begin_of_text|><|ifm|im_start|>user\n<audio>"
    assert suffix == "</audio><|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>"


@pytest.mark.parametrize(("raw", "kept"), [
    ("[SA] خلنا نراجع الـ budget قبل الاجتماع.", "خلنا نراجع الـ budget قبل الاجتماع."),
    ("[msa]الوقت ضيق.", "الوقت ضيق."),
    ("  [AE] [Code-switch] زين", "زين"),
    ("[IFM-NoTranscript]", ""),
    ("[ifm-notranscript] شكرا", ""),
    ("[SA]", ""),
    ("", ""),
    ("قال [باللغة] شي", "قال [باللغة] شي"),
    ("Meeting at three.", "Meeting at three."),
])
def test_only_the_words_after_the_decision_tag_are_kept(raw: str, kept: str) -> None:
    assert without_decision_tag(raw) == kept


def test_a_long_transcript_keeps_every_word_after_the_tag() -> None:
    words = " ".join(["نراجع الأرقام مع الفريق"] * 2000)

    assert without_decision_tag("[SA] " + words) == words


class _PromptedQasr:
    def __init__(self, transcript: str) -> None:
        self._transcript = transcript.encode()
        self.prefix: list[int] = []

    def qasr_transcribe_prompted(self, handle, samples, count, prefix, prefix_len, suffix, suffix_len, buffer, size) -> int:  # noqa: ANN001
        self.prefix = [prefix[i] for i in range(prefix_len)]
        ctypes.memmove(buffer, self._transcript, len(self._transcript))
        return len(self._transcript)

    def qasr_stats(self, handle) -> bytes:  # noqa: ANN001
        return b"{}"


def test_the_native_jais_engine_types_the_transcript_without_its_tag_and_ignores_names_to_spell(tmp_path: Path) -> None:
    adapter = NativeJaisAsrAdapter(tmp_path)
    lib = _PromptedQasr("[SA] أبي أرسل التقرير بكرة.")
    adapter._lib, adapter._handle = lib, 1  # type: ignore[assignment]
    adapter._tokenizer = lambda text: [ord(c) for c in text]

    result = adapter.transcribe_prompted(
        AudioSegment(np.zeros(16000, np.float32), 16000), "Names and terms that may come up: Northwind.", "ar",
    )

    assert result.text == "أبي أرسل التقرير بكرة." and result.language == "ar"
    assert lib.prefix == [ord(c) for c in prompt_text(code_switch=True)[0]]
    assert isinstance(adapter, PromptedTranscriber)
    assert adapter._packed_path() == tmp_path / ".qasr" / "jais-asr.bin"


@pytest.mark.parametrize(("raw", "typed"), [
    ("[SA] إيه اشرح لي وضّح لي", "إيه اشرح لي وضح لي"),
    ("[msa] كَتَبَ الطَّالِبُ دَرْسًا", "كتب الطالب درسا"),
    ("[AE] تمام bus الساعة ٣", "تمام bus الساعة ٣"),
    ("[IFM-NoTranscript] وَ", ""),
])
def test_jais_asrs_vowel_marks_are_dropped_and_everything_else_is_kept(raw: str, typed: str) -> None:
    assert transcript_text(raw) == typed
