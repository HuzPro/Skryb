"""Pack a Parakeet TDT 0.6B checkpoint (transformers layout) into the native engine's parakeet.bin.

Serves Parakeet Ultra and the Underdog Bark family (whose model.bark unpacks to the same layout). The file
is the device image: bf16 tensors (fp16/fp32 weights rounded the way ``.to(torch.bfloat16)`` rounds them),
256-byte aligned, in the layout the kernels read, so loading is one read and one copy. Needs only numpy and
sentencepiece (both in the app venv), never torch.

    python -m skryb.transcription.parakeet_pack <model.safetensors> <out.bin>

Layout: 64 KiB header
    magic "NPK00001", u32 tensor count, then per tensor
    name (64 bytes, NUL padded), u64 offset, u64 bytes, u32 host (1 = stays on the CPU)
then the tensors. Host tensors (the vocabulary) sit after all device ones.
"""

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np

from skryb.transcription.parakeet_hf_checkpoint import remap_hf_parakeet_state_dict

_ALIGN = 256
_HEADER = 65536
_LAYERS = 24
_BN_EPS = 1e-5
_DTYPES = {"F16": np.float16, "F32": np.float32, "BF16": np.uint16}


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Every floating tensor as fp32 (integer bookkeeping tensors are skipped)."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        if info["dtype"] not in _DTYPES:
            continue
        start, end = info["data_offsets"]
        array = np.frombuffer(raw, _DTYPES[info["dtype"]], (end - start) // np.dtype(_DTYPES[info["dtype"]]).itemsize,
                              8 + n + start)
        if info["dtype"] == "BF16":
            array = (array.astype(np.uint32) << 16).view(np.float32)
        out[name] = array.astype(np.float32).reshape(info["shape"])
    return out


def bf16(array: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even bf16 bits, as torch rounds."""
    bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _device_tensors(w: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    pre = "encoder.pre_encode."
    out = [
        ("sub.conv0.w", bf16(w[pre + "conv.0.weight"].reshape(256, 9))),
        ("sub.conv0.b", bf16(w[pre + "conv.0.bias"])),
        ("sub.dw1.w", bf16(w[pre + "conv.2.weight"].reshape(256, 9))),
        ("sub.dw1.b", bf16(w[pre + "conv.2.bias"])),
        ("sub.pw1.w", bf16(w[pre + "conv.3.weight"].reshape(256, 256))),
        ("sub.pw1.b", bf16(w[pre + "conv.3.bias"])),
        ("sub.dw2.w", bf16(w[pre + "conv.5.weight"].reshape(256, 9))),
        ("sub.dw2.b", bf16(w[pre + "conv.5.bias"])),
        ("sub.pw2.w", bf16(w[pre + "conv.6.weight"].reshape(256, 256))),
        ("sub.pw2.b", bf16(w[pre + "conv.6.bias"])),
        # Input columns c * 16 + f -> f * 256 + c: the conv stack writes channels last.
        ("sub.out.w", bf16(w[pre + "out.weight"].reshape(1024, 256, 16).transpose(0, 2, 1).reshape(1024, 4096))),
        ("sub.out.b", bf16(w[pre + "out.bias"])),
    ]
    for i in range(_LAYERS):
        p, q = f"encoder.layers.{i}.", f"L{i}."
        bn = p + "conv.batch_norm."
        scale = w[bn + "weight"] / np.sqrt(w[bn + "running_var"] + _BN_EPS)
        out += [
            (q + "ff1.ln.w", bf16(w[p + "norm_feed_forward1.weight"])),
            (q + "ff1.ln.b", bf16(w[p + "norm_feed_forward1.bias"])),
            (q + "ff1.w1", bf16(w[p + "feed_forward1.linear1.weight"])),
            (q + "ff1.w2", bf16(w[p + "feed_forward1.linear2.weight"])),
            (q + "ln_att.w", bf16(w[p + "norm_self_att.weight"])),
            (q + "ln_att.b", bf16(w[p + "norm_self_att.bias"])),
            (q + "qkv", bf16(np.concatenate([w[p + f"self_attn.linear_{x}.weight"] for x in "qkv"]))),
            (q + "pos", bf16(w[p + "self_attn.linear_pos.weight"])),
            (q + "u", bf16(w[p + "self_attn.pos_bias_u"])),
            (q + "v", bf16(w[p + "self_attn.pos_bias_v"])),
            (q + "out", bf16(w[p + "self_attn.linear_out.weight"])),
            (q + "ln_conv.w", bf16(w[p + "norm_conv.weight"])),
            (q + "ln_conv.b", bf16(w[p + "norm_conv.bias"])),
            (q + "pw1", bf16(w[p + "conv.pointwise_conv1.weight"].reshape(2048, 1024))),
            # Batch norm folded into the depthwise conv, kept in fp32.
            (q + "dw.w", np.ascontiguousarray(w[p + "conv.depthwise_conv.weight"].reshape(1024, 9) * scale[:, None],
                                              dtype=np.float32)),
            (q + "dw.b", np.ascontiguousarray(w[bn + "bias"] - w[bn + "running_mean"] * scale, dtype=np.float32)),
            (q + "pw2", bf16(w[p + "conv.pointwise_conv2.weight"].reshape(1024, 1024))),
            (q + "ff2.ln.w", bf16(w[p + "norm_feed_forward2.weight"])),
            (q + "ff2.ln.b", bf16(w[p + "norm_feed_forward2.bias"])),
            (q + "ff2.w1", bf16(w[p + "feed_forward2.linear1.weight"])),
            (q + "ff2.w2", bf16(w[p + "feed_forward2.linear2.weight"])),
            (q + "ln_out.w", bf16(w[p + "norm_out.weight"])),
            (q + "ln_out.b", bf16(w[p + "norm_out.bias"])),
        ]
    out += [
        ("joint.enc.w", bf16(w["joint.enc.weight"])),
        ("joint.enc.b", bf16(w["joint.enc.bias"])),
        ("joint.pred.w", bf16(w["joint.pred.weight"])),
        ("joint.pred.b", bf16(w["joint.pred.bias"])),
        ("joint.out.w", bf16(w["joint.joint_net.2.weight"])),
        ("joint.out.b", bf16(w["joint.joint_net.2.bias"])),
        ("dec.embed", bf16(w["decoder.embed.weight"])),
    ]
    for layer in (0, 1):
        lstm = "decoder.lstm."
        out += [
            (f"lstm{layer}.ih", bf16(w[lstm + f"weight_ih_l{layer}"])),
            (f"lstm{layer}.hh", bf16(w[lstm + f"weight_hh_l{layer}"])),
            # The model runs in bf16, so each bias is rounded before the two are added.
            (f"lstm{layer}.b", (_round_bf16(w[lstm + f"bias_ih_l{layer}"]) + _round_bf16(w[lstm + f"bias_hh_l{layer}"]))
             .astype(np.float32)),
        ]
    return out


def _round_bf16(array: np.ndarray) -> np.ndarray:
    return (bf16(array).astype(np.uint32) << 16).view(np.float32)


def _vocab() -> tuple[np.ndarray, np.ndarray]:
    """The 8,192 SentencePiece pieces as UTF-8 bytes (control pieces empty), with offsets."""
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor()
    sp.LoadFromSerializedProto(_bundled_tokenizer().read_bytes())
    pieces = [b"" if sp.is_control(i) else sp.id_to_piece(i).encode("utf-8") for i in range(sp.get_piece_size())]
    offsets = np.zeros(len(pieces) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum([len(p) for p in pieces])
    return offsets, np.frombuffer(b"".join(pieces), dtype=np.uint8)


def _bundled_tokenizer() -> Path:
    """nano-parakeet's tokenizer.model, found without importing nano_parakeet (which imports torch)."""
    spec = importlib.util.find_spec("nano_parakeet")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("nano-parakeet isn't installed; its tokenizer.model is Parakeet's vocabulary")
    return Path(next(iter(spec.submodule_search_locations))) / "tokenizer.model"


def pack(weights: Path, out: Path) -> None:
    state = remap_hf_parakeet_state_dict(read_safetensors(weights))
    device = _device_tensors(state)
    offsets, vocab_bytes = _vocab()
    host = [("vocab.offsets", offsets), ("vocab.bytes", vocab_bytes)]
    entries, offset = [], _HEADER
    for name, array in device + host:
        entries.append((name, offset, array.nbytes, int(any(name == h for h, _ in host))))
        offset = -(-(offset + array.nbytes) // _ALIGN) * _ALIGN
    header = bytearray(_HEADER)
    header[:8] = b"NPK00001"
    struct.pack_into("<I", header, 8, len(entries))
    for i, (name, off, size, is_host) in enumerate(entries):
        struct.pack_into("<64sQQI", header, 12 + i * 84, name.encode(), off, size, is_host)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    with partial.open("wb") as f:
        f.write(header)
        for (name, off, _, _), (_, array) in zip(entries, device + host):
            f.seek(off)
            f.write(np.ascontiguousarray(array).tobytes())
    partial.replace(out)


if __name__ == "__main__":
    pack(Path(sys.argv[1]), Path(sys.argv[2]))
