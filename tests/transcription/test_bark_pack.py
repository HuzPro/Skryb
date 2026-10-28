"""Tests for unpacking Underdog Bark's ``model.bark``.

The files here are built by a small encoder written against the same
format, using constriction's own ANS coder, so the decoder is checked
against weights it has never seen, not against itself.
"""

import json
import lzma
from pathlib import Path

import numpy as np
import pytest

from skryb.core.exceptions import TranscriptionError
from skryb.transcription import bark_pack
from skryb.transcription.bark_pack import read_bark_file, unpacked_weights

constriction = pytest.importorskip("constriction")

_CLASSES = 16


def _model(counts: np.ndarray) -> object:
    return constriction.stream.model.Categorical(counts / counts.sum(), perfect=False)


def _ans_words(symbol_runs: list[tuple[np.ndarray, np.ndarray]]) -> bytes:
    """Words a stack decoder reads back run by run, in the order given."""
    coder = constriction.stream.stack.AnsCoder()
    for symbols, counts in reversed(symbol_runs):
        coder.encode_reverse(symbols.astype(np.int32), _model(counts))
    return coder.get_compressed().astype(np.uint32).tobytes()


def _maybe_lzma(blob: bytes, compress: bool) -> bytes:
    return lzma.compress(blob) if compress else blob


def _levels_entry(
    name: str, weights_codes: np.ndarray, levels: np.ndarray, classes: np.ndarray, compress: bool,
) -> tuple[dict, list[bytes]]:
    rows, columns = weights_codes.shape
    level_count = levels.shape[1]
    counts = np.ones((_CLASSES, 2 * level_count + 1), np.uint32)
    runs = []
    for column_class in range(_CLASSES):
        indices = np.nonzero(classes == column_class)[0]
        symbols = weights_codes[:, indices].T.reshape(-1)
        np.add.at(counts[column_class], symbols, 1)
        if len(indices):
            runs.append((symbols, counts[column_class].astype(np.float64)))
    packed_classes = np.zeros((columns + 1) // 2, np.uint8)
    padded = np.concatenate([classes, np.zeros(columns % 2, np.uint8)]).astype(np.uint8)
    packed_classes[:] = (padded[0::2] << 4) | padded[1::2]
    level_blob = _maybe_lzma(levels.astype(np.float16).tobytes(), compress)
    meta = packed_classes.tobytes() + counts.tobytes()
    words = _ans_words(runs)
    entry = {
        "n": name, "k": "levels2", "shape": [rows, columns], "K": level_count,
        "lev": len(level_blob), "zl": compress, "meta": len(meta), "b": len(words),
    }
    return entry, [level_blob, meta, words]


def _scaled_entry(
    name: str, integers: np.ndarray, scale: np.ndarray, bits: int, compress: bool,
) -> tuple[dict, list[bytes]]:
    offset = 2 ** (bits - 1) - 1
    symbols = (integers + offset).reshape(-1)
    counts = np.ones(2 ** bits, np.uint32)
    np.add.at(counts, symbols, 1)
    scale_blob = _maybe_lzma(scale.astype(np.float16).tobytes(), compress)
    words = _ans_words([(symbols, counts.astype(np.float64))])
    entry = {
        "n": name, "k": f"ansint{bits}", "shape": list(integers.shape),
        "s": len(scale_blob), "zs": compress, "meta": counts.nbytes, "b": len(words),
    }
    return entry, [scale_blob, counts.tobytes(), words]


def _raw_entry(name: str, tensor: np.ndarray, compress: bool) -> tuple[dict, list[bytes]]:
    dtype = {np.dtype(np.int64): "torch.int64"}.get(tensor.dtype, "torch.float16")
    blob = _maybe_lzma(tensor.tobytes(), compress)
    entry = {"n": name, "k": "raw", "shape": list(tensor.shape), "dtype": dtype, "b": len(blob), "zr": compress}
    return entry, [blob]


def _write_bark(path: Path, entries: list[tuple[dict, list[bytes]]], fmt: str) -> None:
    header = json.dumps({"format": fmt, "config": "{}", "index": [e for e, _ in entries]}).encode()
    body = b"".join(blob for _, blobs in entries for blob in blobs)
    path.write_bytes(len(header).to_bytes(8, "little") + header + body)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20261028)


def _encoder_weights(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A 1024 x 1537 linear weight: zeros, both signs, every class but one, an odd width."""
    rows, columns, level_count = 1024, 1537, 7
    levels = np.sort(rng.uniform(1e-3, 0.2, (rows, level_count)), axis=1).astype(np.float16)
    codes = rng.integers(0, 2 * level_count + 1, (rows, columns)).astype(np.uint8)
    codes[:, :5] = 0
    classes = rng.integers(0, _CLASSES - 1, columns).astype(np.uint8)
    table = np.concatenate([np.zeros((rows, 1), np.float16), levels, -levels], axis=1)
    expected = np.take_along_axis(table, codes.astype(np.intp), axis=1)
    return codes, levels, classes, expected


@pytest.mark.parametrize(("fmt", "compress"), [("bark-levels-v2", False), ("bark-levels-v3", True)])
def test_every_kind_of_tensor_decodes_to_the_exact_weights_that_were_packed(
    tmp_path: Path, rng: np.random.Generator, fmt: str, compress: bool,
) -> None:
    codes, levels, classes, expected_linear = _encoder_weights(rng)
    conv = rng.integers(-15, 16, (64, 1, 9)).astype(np.int64)
    conv_scale = rng.uniform(1e-3, 1e-1, 64).astype(np.float16)
    lstm = rng.integers(-31, 32, (2560, 640)).astype(np.int64)
    lstm_scale = rng.uniform(1e-3, 1e-1, 2560).astype(np.float16)
    bias = rng.normal(size=1024).astype(np.float16)
    path = tmp_path / "model.bark"
    _write_bark(path, [
        _levels_entry("encoder.layers.0.feed_forward1.linear1.weight", codes, levels, classes, compress),
        _scaled_entry("encoder.layers.0.conv.depthwise_conv.weight", conv, conv_scale, 5, compress),
        _scaled_entry("decoder.lstm.weight_ih_l0", lstm, lstm_scale, 6, compress),
        _raw_entry("encoder.layers.0.feed_forward1.linear1.bias", bias, compress),
        _raw_entry("decoder.embedding.padding_idx", np.array([8192], np.int64), compress),
    ], fmt)

    tensors = read_bark_file(path)

    np.testing.assert_array_equal(tensors["encoder.layers.0.feed_forward1.linear1.weight"], expected_linear)
    assert tensors["encoder.layers.0.feed_forward1.linear1.weight"].dtype == np.float16
    np.testing.assert_array_equal(
        tensors["encoder.layers.0.conv.depthwise_conv.weight"],
        conv.astype(np.float32) * conv_scale.astype(np.float32)[:, None, None],
    )
    np.testing.assert_array_equal(
        tensors["decoder.lstm.weight_ih_l0"],
        lstm.astype(np.float32) * lstm_scale.astype(np.float32)[:, None],
    )
    np.testing.assert_array_equal(tensors["encoder.layers.0.feed_forward1.linear1.bias"], bias)
    assert tensors["decoder.embedding.padding_idx"].tolist() == [8192]


def test_batch_norm_step_counters_are_put_back_for_a_strict_load(tmp_path: Path) -> None:
    path = tmp_path / "model.bark"
    mean = np.zeros(1024, np.float16)
    _write_bark(path, [
        _raw_entry("encoder.layers.0.conv.norm.running_mean", mean, compress=False),
        _raw_entry("encoder.layers.23.conv.norm.running_mean", mean, compress=False),
    ], "bark-levels-v3")

    tensors = read_bark_file(path)

    assert tensors["encoder.layers.0.conv.norm.num_batches_tracked"].dtype == np.int64
    assert tensors["encoder.layers.23.conv.norm.num_batches_tracked"].shape == ()
    assert len(tensors) == 4


def test_an_unknown_format_is_refused_with_its_name(tmp_path: Path) -> None:
    path = tmp_path / "model.bark"
    _write_bark(path, [], "bark-levels-v9")

    with pytest.raises(TranscriptionError, match="bark-levels-v9"):
        read_bark_file(path)


def test_the_weights_are_unpacked_once_and_reused_after(tmp_path: Path, monkeypatch) -> None:
    _write_bark(tmp_path / "model.bark", [
        _raw_entry("joint.head.bias", np.arange(8193, dtype=np.float16), compress=True),
    ], "bark-levels-v3")

    first = unpacked_weights(tmp_path)
    monkeypatch.setattr(bark_pack, "read_bark_file", lambda path: pytest.fail("unpacked twice"))
    second = unpacked_weights(tmp_path)

    from safetensors.numpy import load_file

    assert first == second == tmp_path / ".bark" / "model.safetensors"
    assert load_file(str(first))["joint.head.bias"][-1] == np.float16(8192)
    assert [p.name for p in first.parent.iterdir()] == ["model.safetensors"]


def test_a_failed_unpack_leaves_nothing_that_looks_finished(tmp_path: Path, monkeypatch) -> None:
    _write_bark(tmp_path / "model.bark", [
        _raw_entry("joint.head.bias", np.zeros(4, np.float16), compress=False),
    ], "bark-levels-v3")

    def full_disk(tensors, filename):
        Path(filename).write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr("safetensors.numpy.save_file", full_disk)

    with pytest.raises(OSError):
        unpacked_weights(tmp_path)

    assert not (tmp_path / ".bark" / "model.safetensors").exists()
