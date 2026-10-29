"""Pack a Qwen3-ASR-1.7B-hf checkpoint into the native engine's qasr.bin.

The file is the device image: every tensor 256-byte aligned and already in
the layout the kernels read, so loading is one stream from disk to VRAM.
Needs only numpy and tokenizers (both in the app venv). The adapter packs
once, on first load; by hand:

    python -m skryb.transcription.qwen3_asr_pack <hf_snapshot_dir> <out.bin>

Layout: 64 KiB header
    magic "QASR0001", u32 tensor count, then per tensor
    name (64 bytes, NUL padded), u64 offset, u64 bytes, u32 host (1 = stays on the CPU)
then the tensors. Host tensors (vocab, prompt ids) sit after all device ones.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

_ALIGN = 256
_HEADER = 65536
_ENCODER_LAYERS = 24
_DECODER_LAYERS = 28


def _read_safetensors(path: Path) -> dict[str, np.ndarray]:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    header.pop("__metadata__", None)
    base = 8 + n
    out = {}
    for name, info in header.items():
        assert info["dtype"] == "BF16", name
        start, end = info["data_offsets"]
        out[name] = np.frombuffer(raw, np.uint16, (end - start) // 2, base + start).reshape(info["shape"])
    return out


def _device_tensors(w: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    a = "model.audio_tower."
    t = [
        ("enc.conv1.w", w[a + "conv2d1.weight"].reshape(480, 9)),
        ("enc.conv1.b", w[a + "conv2d1.bias"]),
        # [co][ci][ky][kx] -> [co][ky][kx][ci]: the conv kernel reads 8 channels at a time.
        ("enc.conv2.w", w[a + "conv2d2.weight"].transpose(0, 2, 3, 1)),
        ("enc.conv2.b", w[a + "conv2d2.bias"]),
        ("enc.conv3.w", w[a + "conv2d3.weight"].transpose(0, 2, 3, 1)),
        ("enc.conv3.b", w[a + "conv2d3.bias"]),
        # Input columns c*16+f -> f*480+c, matching the conv3 output layout [t][f][c].
        ("enc.conv_out.w", w[a + "conv_out.weight"].reshape(1024, 480, 16).transpose(0, 2, 1)),
        ("enc.ln_post.w", w[a + "ln_post.weight"]),
        ("enc.ln_post.b", w[a + "ln_post.bias"]),
    ]
    for i in range(_ENCODER_LAYERS):
        p = f"{a}layers.{i}."
        s = p + "self_attn."
        t += [
            (f"enc.{i}.ln1.w", w[p + "self_attn_layer_norm.weight"]),
            (f"enc.{i}.ln1.b", w[p + "self_attn_layer_norm.bias"]),
            (f"enc.{i}.qkv.w", np.concatenate([w[s + "q_proj.weight"], w[s + "k_proj.weight"], w[s + "v_proj.weight"]])),
            (f"enc.{i}.qkv.b", np.concatenate([w[s + "q_proj.bias"], w[s + "k_proj.bias"], w[s + "v_proj.bias"]])),
            (f"enc.{i}.out.w", w[s + "out_proj.weight"]),
            (f"enc.{i}.out.b", w[s + "out_proj.bias"]),
            (f"enc.{i}.ln2.w", w[p + "final_layer_norm.weight"]),
            (f"enc.{i}.ln2.b", w[p + "final_layer_norm.bias"]),
            (f"enc.{i}.fc1.w", w[p + "fc1.weight"]),
            (f"enc.{i}.fc1.b", w[p + "fc1.bias"]),
            (f"enc.{i}.fc2.w", w[p + "fc2.weight"]),
            (f"enc.{i}.fc2.b", w[p + "fc2.bias"]),
        ]
    m = "model.multi_modal_projector."
    t += [
        ("proj.1.w", w[m + "linear_1.weight"]), ("proj.1.b", w[m + "linear_1.bias"]),
        ("proj.2.w", w[m + "linear_2.weight"]), ("proj.2.b", w[m + "linear_2.bias"]),
    ]
    d = "model.language_model."
    t += [("dec.embed", w[d + "embed_tokens.weight"]), ("dec.norm", w[d + "norm.weight"])]
    for i in range(_DECODER_LAYERS):
        p = f"{d}layers.{i}."
        s = p + "self_attn."
        t += [
            (f"dec.{i}.ln1", w[p + "input_layernorm.weight"]),
            (f"dec.{i}.qkv", np.concatenate([w[s + "q_proj.weight"], w[s + "k_proj.weight"], w[s + "v_proj.weight"]])),
            (f"dec.{i}.qnorm", w[s + "q_norm.weight"]),
            (f"dec.{i}.knorm", w[s + "k_norm.weight"]),
            (f"dec.{i}.o", w[s + "o_proj.weight"]),
            (f"dec.{i}.ln2", w[p + "post_attention_layernorm.weight"]),
            (f"dec.{i}.gateup", np.concatenate([w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]])),
            (f"dec.{i}.down", w[p + "mlp.down_proj.weight"]),
        ]
    return t


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode, which byte-level BPE token strings use."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _vocab(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Token bytes as (offsets[n+1] u32, blob u8). Special tokens decode to nothing."""
    tok = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    decoder = _byte_decoder()
    pieces: dict[int, bytes] = {i: bytes(decoder[c] for c in s) for s, i in tok["model"]["vocab"].items()}
    for added in tok["added_tokens"]:
        pieces[added["id"]] = b"" if added["special"] else added["content"].encode()
    n = 151936
    offsets = np.zeros(n + 1, np.uint32)
    blob = bytearray()
    for i in range(n):
        offsets[i] = len(blob)
        blob += pieces.get(i, b"")
    offsets[n] = len(blob)
    return offsets, np.frombuffer(bytes(blob), np.uint8)


def _prompt_ids(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Token ids around the audio placeholders, as the chat template renders them."""
    from tokenizers import Tokenizer

    from skryb.transcription.qwen3_asr_prompt import prompt_text

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prefix, suffix = prompt_text(None, "en")
    enc =lambda s: np.array(tok.encode(s, add_special_tokens=False).ids, np.int32)
    return enc(prefix), enc(suffix)


def pack(model_dir: Path, out: Path) -> None:
    """Write *out* from the snapshot in *model_dir*, via a temporary name renamed into place."""
    tensors = _device_tensors(_read_safetensors(model_dir / "model.safetensors"))
    offsets, blob = _vocab(model_dir)
    prefix, suffix = _prompt_ids(model_dir)
    host = [("vocab.offsets", offsets), ("vocab.bytes", blob), ("prompt.prefix", prefix), ("prompt.suffix", suffix)]
    entries = [(n, a, 0) for n, a in tensors] + [(n, a, 1) for n, a in host]
    assert 8 + 4 + len(entries) * 84 <= _HEADER
    table, pos = [], _HEADER
    for name, arr, is_host in entries:
        table.append((name, pos, arr.nbytes, is_host))
        pos += -(-arr.nbytes // _ALIGN) * _ALIGN
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with partial.open("wb") as f:
        head = b"QASR0001" + struct.pack("<I", len(entries))
        for name, offset, nbytes, is_host in table:
            head += name.encode().ljust(64, b"\0") + struct.pack("<QQI", offset, nbytes, is_host)
        f.write(head.ljust(_HEADER, b"\0"))
        for (_, arr, _), (_, offset, _, _) in zip(entries, table):
            f.seek(offset)
            f.write(np.ascontiguousarray(arr).tobytes())
        f.truncate(pos)
    partial.replace(out)


if __name__ == "__main__":
    pack(Path(sys.argv[1]), Path(sys.argv[2]))
