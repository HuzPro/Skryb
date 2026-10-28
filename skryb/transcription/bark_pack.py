"""Unpack Underdog Bark's ``model.bark`` into a transformers-layout safetensors file.

A ``.bark`` file (formats ``bark-levels-v2`` and ``-v3``) holds the weights
of a stock transformers ``ParakeetForTDT``, entropy coded with ANS:

- encoder linear weights: per-row fp16 levels plus one code per weight,
  with sixteen column classes per tensor, each with its own symbol model;
- other matrices: per-row symmetric 5- or 6-bit integers with fp16 scales;
- small tensors: fp16.

v3 adds LZMA on the small side blobs. Decoding is exact. This reads the
format the way Conway Research's own ``bark_loader.py`` does, without
running that script.

The result is written once, beside the download (``snapshots/<sha>/.bark``),
the way Phonon-2's archive is unpacked, and renamed into place last so an
interrupted unpack never looks finished. The levels are fp16 values, so
those tensors are stored as fp16 without losing a bit.
"""

import json
import logging
import lzma
import os
import time
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.transcription.backend_imports import make_engine_dependency_error


_logger = logging.getLogger(__name__)

BARK_FILENAME = "model.bark"
# Dot-prefixed so it can never collide with a directory the upstream repo
# ships at the pinned commit.
_UNPACKED_DIRECTORY_NAME = ".bark"
_WEIGHTS_FILENAME = "model.safetensors"
_FORMATS = frozenset({"bark-levels-v2", "bark-levels-v3"})
_COLUMN_CLASSES = 16
_HEADER_LENGTH_BYTES = 8
_RAW_DTYPES = {"torch.int64": np.int64, "torch.int32": np.int32}
_INSTALL_HINT = (
    "Underdog Bark needs the constriction package to unpack its weights. "
    "Install it with: pip install constriction==0.5.0"
)


def unpacked_weights(snapshot: Path) -> Path:
    """The safetensors file unpacked from *snapshot*'s ``model.bark``, made once."""
    weights = snapshot / _UNPACKED_DIRECTORY_NAME / _WEIGHTS_FILENAME
    if weights.is_file():
        return weights
    started_at = time.monotonic()
    _logger.info("Unpacking %s into %s", BARK_FILENAME, weights.parent)
    tensors = read_bark_file(snapshot / BARK_FILENAME)
    _save_atomically(tensors, weights)
    _logger.info("Unpacked %s in %.1fs", BARK_FILENAME, time.monotonic() - started_at)
    return weights


def read_bark_file(path: Path) -> dict[str, np.ndarray]:
    """Every tensor in *path*, under transformers ``ParakeetForTDT`` names."""
    constriction = _import_constriction()
    raw = path.read_bytes()
    header_end = _HEADER_LENGTH_BYTES + int.from_bytes(raw[:_HEADER_LENGTH_BYTES], "little")
    header = json.loads(raw[_HEADER_LENGTH_BYTES:header_end])
    if header.get("format") not in _FORMATS:
        raise TranscriptionError(f"{path.name} is in an unknown format: {header.get('format')!r}")
    blobs = _Blobs(raw, header_end)
    tensors = {
        entry["n"]: _decode_entry(constriction, entry, blobs)
        for entry in header["index"]
    }
    return _with_batch_norm_counters(tensors)


class _Blobs:
    """Reads the file's blobs in order, inflating the LZMA ones."""

    def __init__(self, raw: bytes, offset: int) -> None:
        self._raw = raw
        self._offset = offset

    def take(self, length: int, compressed: bool = False) -> bytes:
        blob = self._raw[self._offset:self._offset + length]
        self._offset += length
        return lzma.decompress(blob) if compressed else blob


def _decode_entry(constriction: object, entry: dict, blobs: _Blobs) -> np.ndarray:
    kind = entry["k"]
    shape = tuple(entry["shape"])
    if kind == "levels2":
        return _decode_levels(constriction, entry, shape, blobs)
    if kind.startswith("ansint"):
        return _decode_scaled_integers(constriction, entry, shape, int(kind[6:]), blobs)
    if kind == "raw":
        dtype = _RAW_DTYPES.get(entry["dtype"], np.float16)
        return np.frombuffer(blobs.take(entry["b"], entry.get("zr")), dtype).reshape(shape).copy()
    raise TranscriptionError(f"Unknown tensor encoding {kind!r} in {BARK_FILENAME}")


def _decode_levels(
    constriction: object, entry: dict, shape: tuple[int, ...], blobs: _Blobs,
) -> np.ndarray:
    """One code per weight: 0 is zero, 1..K a row's level, K+1..2K its negation."""
    level_count, rows = entry["K"], shape[0]
    columns = int(np.prod(shape[1:]))
    levels = np.frombuffer(blobs.take(entry["lev"], entry.get("zl")), np.float16)
    levels = levels.reshape(rows, level_count)
    meta = blobs.take(entry["meta"])
    class_bytes = (columns * 4 + 7) // 8
    packed = np.frombuffer(meta[:class_bytes], np.uint8)
    column_classes = np.stack([packed >> 4, packed & 0x0F], axis=1).reshape(-1)[:columns]
    counts = np.frombuffer(meta[class_bytes:], np.uint32).astype(np.float64)
    counts = counts.reshape(_COLUMN_CLASSES, 2 * level_count + 1)
    words = np.frombuffer(blobs.take(entry["b"]), np.uint32).copy()
    codes = _decode_codes(constriction, column_classes, counts, words, rows, columns)
    table = np.concatenate([np.zeros((rows, 1), np.float16), levels, -levels], axis=1)
    return np.take_along_axis(table, codes.astype(np.intp), axis=1).reshape(shape)


def _decode_codes(
    constriction: object,
    column_classes: np.ndarray,
    counts: np.ndarray,
    words: np.ndarray,
    rows: int,
    columns: int,
) -> np.ndarray:
    """Class by class, each class's columns were coded column-major."""
    decoder = constriction.stream.stack.AnsCoder(words)
    codes = np.empty((rows, columns), np.uint8)
    for column_class, class_counts in enumerate(counts):
        indices = np.nonzero(column_classes == column_class)[0]
        if not len(indices):
            continue
        model = constriction.stream.model.Categorical(
            class_counts / class_counts.sum(), perfect=False,
        )
        symbols = decoder.decode(model, rows * len(indices))
        codes[:, indices] = symbols.reshape(len(indices), rows).T
    return codes


def _decode_scaled_integers(
    constriction: object, entry: dict, shape: tuple[int, ...], bits: int, blobs: _Blobs,
) -> np.ndarray:
    scale = np.frombuffer(blobs.take(entry["s"], entry.get("zs")), np.float16).astype(np.float32)
    counts = np.frombuffer(blobs.take(entry["meta"]), np.uint32).astype(np.float64)
    words = np.frombuffer(blobs.take(entry["b"]), np.uint32).copy()
    model = constriction.stream.model.Categorical(counts / counts.sum(), perfect=False)
    symbols = constriction.stream.stack.AnsCoder(words).decode(model, int(np.prod(shape)))
    integers = (symbols - (2 ** (bits - 1) - 1)).astype(np.float32)
    return (integers.reshape(len(scale), -1) * scale[:, None]).reshape(shape)


def _with_batch_norm_counters(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Put back the BatchNorm step counters the file leaves out.

    Inference never reads them, but a strict load wants every buffer, and
    every other transformers checkpoint carries them.
    """
    counters = {
        name.removesuffix("running_mean") + "num_batches_tracked": np.zeros((), np.int64)
        for name in tensors
        if name.endswith(".running_mean")
    }
    return {**counters, **tensors}


def _save_atomically(tensors: dict[str, np.ndarray], destination: Path) -> None:
    from safetensors.numpy import save_file

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".partial")
    save_file(tensors, str(staging))
    os.replace(staging, destination)


def _import_constriction() -> object:
    try:
        import constriction
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="constriction", install_hint=_INSTALL_HINT, underlying=exc,
        ) from exc
    return constriction
