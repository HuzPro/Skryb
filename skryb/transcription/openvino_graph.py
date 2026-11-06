"""Pieces shared by the OpenVINO graphs built straight from ``model.safetensors``, without torch.

Weights are read one tensor at a time from a memory map. Linear layers are either dense (fp32 here, fp16 once
saved, for the NPU) or int8 with one scale per output row, as int8 -> Convert -> Multiply, the form OpenVINO
keeps compressed and expands as it multiplies.
"""

import importlib
import json
import os
import struct
from pathlib import Path

import numpy as np

LAYER_NORM_EPS = 1e-5


class Safetensors:
    """The tensors of one ``.safetensors`` file, each read from a memory map when asked for."""

    def __init__(self, path: Path) -> None:
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(size))
        self.header.pop("__metadata__", None)
        self._data = np.memmap(path, np.uint8, mode="r", offset=8 + size)

    def shape(self, name: str) -> list[int]:
        return list(self.header[name]["shape"])

    def bf16_rows(self, name: str) -> np.ndarray:
        """A bf16 tensor's raw bits as uint16, still memory-mapped: rows are widened only when looked up."""
        info = self.header[name]
        if info["dtype"] != "BF16":
            raise ValueError(f"{name} is {info['dtype']}, not BF16")
        start, end = info["data_offsets"]
        return self._data[start:end].view(np.uint16).reshape(info["shape"])

    def __getitem__(self, name: str) -> np.ndarray:
        info = self.header[name]
        start, end = info["data_offsets"]
        raw = self._data[start:end]
        if info["dtype"] == "BF16":
            values = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            values = raw.view({"F32": np.float32, "F16": np.float16}[info["dtype"]]).astype(np.float32)
        return values.reshape(info["shape"])


class ShardedSafetensors:
    """A checkpoint split over several ``.safetensors`` shards, read like one."""

    def __init__(self, paths: list[Path]) -> None:
        parts = [Safetensors(path) for path in paths]
        self._where = {name: part for part in parts for name in part.header}

    def shape(self, name: str) -> list[int]:
        return self._where[name].shape(name)

    def bf16_rows(self, name: str) -> np.ndarray:
        return self._where[name].bf16_rows(name)

    def __getitem__(self, name: str) -> np.ndarray:
        return self._where[name][name]


def open_checkpoint(snapshot: Path) -> "Safetensors | ShardedSafetensors":
    """The snapshot's one ``model.safetensors``, or the shards its index names."""
    index = snapshot / "model.safetensors.index.json"
    if not index.exists():
        return Safetensors(snapshot / "model.safetensors")
    shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    return ShardedSafetensors([snapshot / shard for shard in shards])


def widen_bf16(rows: np.ndarray) -> np.ndarray:
    """bf16 bits (uint16) to float32, exactly."""
    return (np.asarray(rows, np.uint32) << 16).view(np.float32)


def save(openvino: object, model: object, path: Path, compress_to_fp16: bool) -> None:
    staged = path.with_name(path.stem + ".partial.xml")
    openvino.save_model(model, str(staged), compress_to_fp16=compress_to_fp16)
    os.replace(staged.with_suffix(".bin"), path.with_suffix(".bin"))
    os.replace(staged, path)


def opset(openvino: object) -> object:
    return importlib.import_module(f"{openvino.__name__}.opset13")


def _int8_rows(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.abs(weight).max(axis=1, keepdims=True) / 127
    scale = np.where(scale == 0, 1, scale).astype(np.float32)
    return np.clip(np.round(weight / scale), -127, 127).astype(np.int8), scale


class Graph:
    """The opset and the weights, so each layer reads as the model's own forward pass."""

    def __init__(self, openvino: object, weights: Safetensors) -> None:
        self.ov = openvino
        self.ops = opset(openvino)
        self.weights = weights

    def const(self, value: object) -> object:
        return self.ops.constant(value)

    def int8_linear(self, x: object, name: str) -> object:
        q, scale = _int8_rows(self.weights[name])
        w = self.ops.multiply(self.ops.convert(self.const(q), self.ov.Type.f32), self.const(scale))
        return self.ops.matmul(x, w, transpose_a=False, transpose_b=True)

    def dense(self, x: object, name: str, bias: bool = True) -> object:
        y = self.ops.matmul(x, self.const(self.weights[name + ".weight"]), transpose_a=False, transpose_b=True)
        return self.ops.add(y, self.const(self.weights[name + ".bias"])) if bias else y

    def rms(self, x: object, name: str, eps: float, groups: int = 1) -> object:
        """RMSNorm over the last axis, or over each of its *groups* equal slices on its own."""
        ops = self.ops
        width = self.weights.shape(name)[0]
        grouped = ops.reshape(x, self.const([0, 0, groups, width // groups]), True) if groups > 1 else x
        variance = ops.reduce_mean(ops.multiply(grouped, grouped), self.const([-1]), keep_dims=True)
        normed = ops.divide(grouped, ops.sqrt(ops.add(variance, self.const(np.float32(eps)))))
        if groups > 1:
            normed = ops.reshape(normed, self.const([0, 0, width]), True)
        return ops.multiply(normed, self.const(self.weights[name]))

    def rotate(self, t: object, cos: object, sin: object, head_dim: int) -> object:
        """Rotary embedding, t * cos + rotate_half(t) * sin, in the Slice form the NPU reads right.

        Built from Split and Negative instead, the NPU's logits drift by up to a quarter of their range.
        """
        ops, half = self.ops, head_dim // 2
        first = ops.slice(t, self.const([0]), self.const([half]), self.const([1]), self.const([-1]))
        second = ops.slice(t, self.const([half]), self.const([head_dim]), self.const([1]), self.const([-1]))
        rotated = ops.concat([ops.multiply(second, self.const(np.float32(-1))), first], -1)
        return ops.add(ops.multiply(t, cos), ops.multiply(rotated, sin))

    def layer_norm(self, x: object, name: str) -> object:
        ops = self.ops
        normed = ops.mvn(x, self.const([-1]), True, LAYER_NORM_EPS, "inside_sqrt")
        return ops.add(ops.multiply(normed, self.const(self.weights[name + ".weight"])), self.const(self.weights[name + ".bias"]))

    def parameter(self, shape: list[int], element_type: object, name: str) -> object:
        p = self.ops.parameter(shape, element_type, name=name)
        p.output(0).get_tensor().set_names({name})
        return p
