"""Load the streaming Parakeet ``.nemo`` (weights + tokenizer) without NeMo.

A ``.nemo`` is a tar of ``model_config.yaml``, ``model_weights.ckpt`` and a
SentencePiece ``tokenizer.model``. We extract the ckpt to disk and
memory-map it (loading the 2.4GB blob fully into RAM crashes), remap NeMo
key names onto :mod:`...cache_aware.model`'s module hierarchy, and load the
bundled tokenizer.
"""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import torch

_CKPT_SUFFIX = "model_weights.ckpt"
_TOKENIZER_SUFFIX = "tokenizer.model"
_PREDICTION_PREFIX = "decoder.prediction."
_LSTM_PREFIX = "decoder.prediction.dec_rnn.lstm."


def _extract_member(nemo_path: Path, suffix: str, dest_dir: Path) -> Path:
    with tarfile.open(nemo_path, "r:*") as tar:
        name = next((n for n in tar.getnames() if n.endswith(suffix)), None)
        if name is None:
            raise FileNotFoundError(f"{suffix} not found in {nemo_path}")
        dest = dest_dir / Path(name).name
        src = tar.extractfile(name)
        with open(dest, "wb") as out:
            while True:
                block = src.read(1 << 20)
                if not block:
                    break
                out.write(block)
    return dest


def _torch_load(ckpt_path: Path, map_location: str, weights_only: bool) -> object:
    return torch.load(
        str(ckpt_path), map_location=map_location, weights_only=weights_only,
    )


def _load_checkpoint(ckpt_path: Path, map_location: str) -> object:
    """Load the ckpt, trying ``weights_only=True`` then falling back to False.

    No mmap: on Windows, reading data from an mmap-backed torch.load tensor
    segfaults. ``map_location`` streams each storage straight to the target
    device as it deserializes, so the CPU never holds the whole ~2.4GB blob at
    once (the box has little free RAM); GPU has ample VRAM.
    """
    for weights_only in (True, False):
        try:
            return _torch_load(ckpt_path, map_location, weights_only)
        except Exception:  # noqa: BLE001 — try the next load strategy
            continue
    raise RuntimeError(f"could not torch.load {ckpt_path}")


def _raw_state_dict(ckpt_path: Path, map_location: str) -> dict:
    state = _load_checkpoint(ckpt_path, map_location)
    if isinstance(state, dict) and "state_dict" in state:
        return state["state_dict"]
    return state


def remap_state_dict(sd: dict) -> dict:
    """Map NeMo checkpoint key names onto our module hierarchy."""
    remapped: dict = {}
    for key, value in sd.items():
        if key == "preprocessor.featurizer.window":
            remapped["preprocessor.window"] = value
        elif key == "preprocessor.featurizer.fb":
            remapped["preprocessor.fb"] = value
        elif key == "decoder.prediction.embed.weight":
            remapped["decoder.embed.weight"] = value
        elif key.startswith(_LSTM_PREFIX):
            remapped["decoder.lstm." + key[len(_LSTM_PREFIX):]] = value
        elif ".conv.batch_norm." in key:
            remapped[key.replace(".conv.batch_norm.", ".conv.norm.")] = value
        elif key.startswith(_PREDICTION_PREFIX):
            # Any other prediction-net params keep their tail under decoder.*
            remapped["decoder." + key[len(_PREDICTION_PREFIX):]] = value
        else:
            remapped[key] = value
    return remapped


def load_streaming_state_dict(
    nemo_path: str | Path, map_location: str = "cpu",
) -> dict:
    nemo_path = Path(nemo_path)
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _extract_member(nemo_path, _CKPT_SUFFIX, Path(tmp))
        raw = _raw_state_dict(ckpt, map_location)
    return remap_state_dict(raw)


def load_tokenizer(nemo_path: str | Path) -> object:
    import sentencepiece as spm

    nemo_path = Path(nemo_path)
    with tempfile.TemporaryDirectory() as tmp:
        model_file = _extract_member(nemo_path, _TOKENIZER_SUFFIX, Path(tmp))
        proto = model_file.read_bytes()
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(proto)
    return sp


def load_streaming_model(
    nemo_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> object:
    """Build the model, load weights + tokenizer, move to device."""
    from voicetotext.transcription.streaming.cache_aware.model import (
        CacheAwareStreamingParakeet,
    )

    # Build on the target device first so the random-init CPU copy is freed
    # before the checkpoint loads; load weights straight to the device too.
    model = CacheAwareStreamingParakeet().to(device).eval()
    sd = load_streaming_state_dict(nemo_path, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    del sd
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"state_dict mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if dtype is not None:
        model.encoder.to(dtype)
        model.decoder.to(dtype)
        model.joint.to(dtype)
    model.sp = load_tokenizer(nemo_path)
    return model
