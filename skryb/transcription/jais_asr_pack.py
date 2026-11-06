"""Pack a Jais-ASR checkpoint into the native engine's qasr.bin, its decoder in 4 bits.

Jais-ASR is Qwen3-ASR's layout grown: the same audio tower, wider and deeper,
an adapter into a larger decoder, and an untied head. The decoder matrices,
token embeddings and head are quantized here, once, so the engine loads about
4 GB instead of the 11.5 GB checkpoint. The format is qwen3_asr_pack's plus a
"config" host entry the engine reads the model's shape from.

    python -m skryb.transcription.jais_asr_pack <hf_snapshot_dir> <out.bin>
"""
import json
import sys
from pathlib import Path

import numpy as np

from skryb.transcription.qwen3_asr_pack import _vocab, encoder_tensors, read_checkpoint, write_pack

# Rows quantized at a time: bounds the float32 working copy of the 250k-row embedding to ~170 MB.
_ROW_BLOCK = 16384


def quantize_q4(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """bf16 bits [N][K] as 4-bit nibbles [N][K/2] (low nibble first) and fp16 scales [N][K/32].

    Bit for bit what the engine's quantize_q4_kernel does: per group of 32,
    scale = fp16(max |w| / 7), q = clamp(rint(w / scale), -8, 7) + 8.
    """
    n, k = bits.shape
    packed = np.empty((n, k // 2), np.uint8)
    scales = np.empty((n, k // 32), np.float16)
    for r0 in range(0, n, _ROW_BLOCK):
        block = np.ascontiguousarray(bits[r0:r0 + _ROW_BLOCK])
        rows = block.shape[0]
        x = (block.astype(np.uint32) << 16).view(np.float32).reshape(rows, k // 32, 32)
        amax = np.abs(x).max(axis=2)
        scale = np.where(amax > 0, amax / np.float32(7), np.float32(1)).astype(np.float16)
        scale[scale == 0] = 1  # a group too small for fp16 quantizes to zeros
        inv = np.float32(1) / scale.astype(np.float32)
        q = (np.clip(np.rint(x * inv[..., None]), -8, 7) + 8).astype(np.uint8)
        packed[r0:r0 + rows] = (q[..., 0::2] | (q[..., 1::2] << 4)).reshape(rows, k // 2)
        scales[r0:r0 + rows] = scale
    return packed, scales


def _q4(name: str, bits: np.ndarray) -> list[tuple[str, np.ndarray]]:
    packed, scales = quantize_q4(bits)
    return [(name + ".q", packed), (name + ".s", scales)]


def engine_config(config: dict) -> np.ndarray:
    """The 18 ints the engine's Cfg reads, in its order; the rope base rides as its float32 bits."""
    audio, text = config["audio_config"], config["text_config"]
    assert text["head_dim"] == 128 and text.get("rope_head_dim", 128) == 128 and not text.get("attention_gate_func")
    assert audio["d_model"] // audio["encoder_attention_heads"] == 64
    rope_theta = text.get("rope_parameters", {}).get("rope_theta", text.get("rope_theta"))
    eos = config["eos_token_id"], 250019  # <|ifm|endoftext|> and <|ifm|im_end|>, which closes the turn
    ints = [
        audio["encoder_layers"], audio["d_model"], audio["encoder_attention_heads"], audio["encoder_ffn_dim"],
        audio["output_dim"], text["num_hidden_layers"], text["hidden_size"], text["num_attention_heads"],
        text["num_key_value_heads"], text["intermediate_size"], text["vocab_size"], eos[0], eos[1],
        int(bool(text["query_key_norm"])), text["layernorm_num_groups"], 1, 1,
        int(np.float32(rope_theta).view(np.int32)),
    ]
    return np.array(ints, np.int32)


def _device_tensors(w: dict[str, np.ndarray], config: dict) -> list[tuple[str, np.ndarray]]:
    audio, text = config["audio_config"], config["text_config"]
    a, m, d = "model.audio_tower.", "model.embed_audio.", "model.language_model."
    t = encoder_tensors(w, audio["encoder_layers"], audio["d_model"])
    t += [
        ("proj.1.w", w[a + "proj1.weight"]), ("proj.1.b", w[a + "proj1.bias"]),
        ("proj.2.w", w[a + "proj2.weight"]), ("proj.2.b", w[a + "proj2.bias"]),
        ("adapter.norm.w", w[m + "norm.weight"]), ("adapter.norm.b", w[m + "norm.bias"]),
        ("adapter.1.w", w[m + "linear_1.weight"]), ("adapter.1.b", w[m + "linear_1.bias"]),
        ("adapter.2.w", w[m + "linear_2.weight"]), ("adapter.2.b", w[m + "linear_2.bias"]),
        ("dec.norm", w[d + "norm.weight"]),
    ]
    t += _q4("dec.embed", w[d + "embed_tokens.weight"]) + _q4("dec.head", w["lm_head.weight"])
    for i in range(text["num_hidden_layers"]):
        p = f"{d}layers.{i}."
        s = p + "self_attn."
        t += [(f"dec.{i}.ln1", w[p + "input_layernorm.weight"]), (f"dec.{i}.ln2", w[p + "post_attention_layernorm.weight"])]
        t += _q4(f"dec.{i}.qkv", np.concatenate([w[s + "q_proj.weight"], w[s + "k_proj.weight"], w[s + "v_proj.weight"]]))
        t += _q4(f"dec.{i}.o", w[s + "o_proj.weight"])
        t += _q4(f"dec.{i}.gateup", np.concatenate([w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]]))
        t += _q4(f"dec.{i}.down", w[p + "mlp.down_proj.weight"])
    return t


def prompt_text(code_switch: bool) -> tuple[str, str]:
    """The text before the audio tokens and after them, as Jais-ASR's chat template renders a transcription turn.

    Code-switch mode keeps English words in English instead of writing them in Arabic letters.
    """
    system = "<|ifm|im_start|>system\n[Code-switch]<|ifm|im_end|>" if code_switch else ""
    return (
        f"<|ifm|begin_of_text|>{system}<|ifm|im_start|>user\n<audio>",
        "</audio><|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>",
    )


def _prompt_ids(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    enc = lambda s: np.array(tok.encode(s, add_special_tokens=False).ids, np.int32)  # noqa: E731
    prefix, suffix = prompt_text(code_switch=True)
    return enc(prefix), enc(suffix)


def pack(model_dir: Path, out: Path) -> None:
    """Write *out* from the snapshot in *model_dir*, via a temporary name renamed into place."""
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tensors = _device_tensors(read_checkpoint(model_dir), config)
    offsets, blob = _vocab(model_dir, config["text_config"]["vocab_size"])
    prefix, suffix = _prompt_ids(model_dir)
    host = [
        ("vocab.offsets", offsets), ("vocab.bytes", blob), ("prompt.prefix", prefix), ("prompt.suffix", suffix),
        ("config", engine_config(config)),
    ]
    write_pack(out, tensors, host)


if __name__ == "__main__":
    pack(Path(sys.argv[1]), Path(sys.argv[2]))
