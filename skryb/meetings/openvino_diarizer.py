"""Nemotron 3 Diarization with its model step on an Intel NPU through OpenVINO, without torch or a GPU.

The step is the model's encoder (31 bidirectional transformer layers), its projection, upsampler and speaker
classifier over the speaker cache, the FIFO queue and the new chunk: at most 264 + 264 + 13 frames. The NPU
compiles fixed shapes, so every step runs at ``STEP_ROWS`` with the rows past the real ones masked out of
attention and zeroed before the upsampler's convolution, which then sees the zero padding it would see at
the true end. The feature stacking, the speaker cache and the labels stay in NumPy, shared with the native
diarizer (``diarization``). The graph is built from ``model.safetensors`` the first time, into
``snapshots/<sha>/.openvino``.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.meetings.diarization import DiarizerStream, SpeakerCache
from skryb.transcription.openvino_graph import Graph, Safetensors, opset, save
from skryb.transcription.openvino_parakeet_adapter import import_openvino

_logger = logging.getLogger(__name__)

GRAPH_FILENAME = "diarizer-step.xml"
EXPORT_DIRECTORY_NAME = ".openvino"
CACHE_DIRECTORY_NAME = "compiled-npu"
# The speaker cache (264), the FIFO queue (264) and a low-latency chunk with its look-ahead (9 + 4), rounded up.
STEP_ROWS = 544
_TOWER = "model.audio_tower."


def build_step(openvino: object, weights: Safetensors, config: dict, rows: int = STEP_ROWS) -> object:
    """embeds [1, rows, 512] and the count of real rows -> speaker logits [rows * 8, num_speakers]."""
    g, ops = Graph(openvino, weights), opset(openvino)
    audio, head = config["audio_config"], config["head_config"]
    width, heads = audio["hidden_size"], audio["num_attention_heads"]
    head_dim = width // heads
    factor = head["subsampling_factor"]
    embeds = g.parameter([1, rows, width], openvino.Type.f32, "embeds")
    valid = g.parameter([1], openvino.Type.i64, "valid")

    theta = audio["rope_parameters"]["rope_theta"]
    inv_freq = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
    freqs = np.arange(rows, dtype=np.float32)[:, None] * inv_freq[None, :]
    angles = np.concatenate([freqs, freqs], axis=-1)
    cos, sin = g.const(np.cos(angles).astype(np.float32)), g.const(np.sin(angles).astype(np.float32))
    real = ops.less(g.const(np.arange(rows, dtype=np.int64)), valid)
    keys = ops.unsqueeze(real, g.const([0, 1, 2]))
    split = lambda t: ops.transpose(ops.reshape(t, g.const([1, rows, heads, head_dim]), False), g.const([0, 2, 1, 3]))

    x = g.layer_norm(embeds, f"{_TOWER}input_layer_norm")
    for i in range(audio["num_hidden_layers"]):
        p = f"{_TOWER}layers.{i}."
        h = g.layer_norm(x, p + "layer_norm1")
        q = g.rotate(split(g.dense(h, p + "self_attn.q_proj", bias=False)), cos, sin, head_dim)
        k = g.rotate(split(g.dense(h, p + "self_attn.k_proj", bias=False)), cos, sin, head_dim)
        v = split(g.dense(h, p + "self_attn.v_proj", bias=False))
        o = ops.scaled_dot_product_attention(q, k, v, keys, causal=False)
        o = ops.reshape(ops.transpose(o, g.const([0, 2, 1, 3])), g.const([1, rows, width]), False)
        x = ops.add(x, g.dense(o, p + "self_attn.o_proj"))
        h = g.layer_norm(x, p + "layer_norm2")
        x = ops.add(x, g.dense(ops.gelu(g.dense(h, p + "mlp.fc1"), "erf"), p + "mlp.fc2"))
    x = g.dense(g.layer_norm(x, f"{_TOWER}layer_norm"), "model.proj")
    x = ops.multiply(x, ops.unsqueeze(ops.convert(real, openvino.Type.f32), g.const([0, 2])))
    x = ops.convolution(ops.transpose(x, g.const([0, 2, 1])), g.const(weights["model.upsampler.conv.weight"]), [1], [1], [1], [1])
    x = ops.add(x, g.const(weights["model.upsampler.conv.bias"].reshape(1, -1, 1)))
    x = ops.reshape(ops.transpose(x, g.const([0, 2, 1])), g.const([rows * factor, head["hidden_size"]]), False)
    x = g.dense(ops.relu(x), "classifier.dense")
    logits = g.dense(ops.relu(x), "classifier.out_proj")
    logits.output(0).get_tensor().set_names({"logits"})
    return openvino.Model([ops.result(logits)], [embeds, valid], "nemotron-diarizer-step")


class OpenVinoDiarizer:
    """Who is talking, one label per 10 ms, with the model step on the NPU (or OpenVINO's CPU device)."""

    frame_seconds = 0.01

    def __init__(self, snapshot: Path) -> None:
        started_at = time.monotonic()
        openvino = import_openvino()
        config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        weights = Safetensors(snapshot / "model.safetensors")
        files = snapshot / EXPORT_DIRECTORY_NAME
        if not (files / GRAPH_FILENAME).is_file():
            files.mkdir(parents=True, exist_ok=True)
            save(openvino, build_step(openvino, weights, config), files / GRAPH_FILENAME, compress_to_fp16=True)
        core = openvino.Core()
        try:
            self._compiled = core.compile_model(str(files / GRAPH_FILENAME), "NPU", {"CACHE_DIR": str(files / CACHE_DIRECTORY_NAME)})
            self.device = "NPU"
        except RuntimeError:
            _logger.warning("The NPU refused the diarizer; it runs on the CPU", exc_info=True)
            self._compiled = core.compile_model(str(files / GRAPH_FILENAME), "CPU")
            self.device = "CPU"
        self._request = self._compiled.create_infer_request()
        self._embedder = weights[f"{_TOWER}embedder.projection.weight"]
        self._silence = weights["silence_embeds"]
        self._factor = int(config["head_config"]["subsampling_factor"])
        self._cache = SpeakerCache(config["streaming_config"])
        self._stream = DiarizerStream(snapshot, self._model_step)
        _logger.info("[load diarizer openvino] on %s in %.2fs", self.device, time.monotonic() - started_at)

    def push(self, audio: np.ndarray, last: bool = False) -> list[int]:
        return self._stream.push(audio, last)

    def dispose(self) -> None:
        self._request = self._compiled = None

    def _model_step(self, features: np.ndarray, lookahead: int) -> np.ndarray:
        factor = self._stream.factor
        padding = -len(features) % factor
        embeds = np.pad(features, ((0, padding), (0, 0))).reshape(-1, factor * features.shape[1]) @ self._embedder.T
        num_chunk_frames = len(embeds) - lookahead
        if num_chunk_frames < 1:
            raise TranscriptionError("a diarizer chunk had no frames to score")
        cached = self._cache.get_embeds()
        step_input = np.concatenate([cached, embeds]).astype(np.float32)
        logits = self.forward(step_input)
        self._cache.update(step_input, logits, self._silence, num_chunk_frames)
        start = len(cached) * factor
        return logits[start:start + num_chunk_frames * factor][:len(features)]

    def forward(self, step_input: np.ndarray) -> np.ndarray:
        rows = len(step_input)
        if rows > STEP_ROWS:
            raise TranscriptionError(f"a diarizer step had {rows} frames, more than the {STEP_ROWS} it was built for")
        padded = np.zeros((1, STEP_ROWS, step_input.shape[1]), np.float32)
        padded[0, :rows] = step_input
        result = self._request.infer({"embeds": padded, "valid": np.array([rows], np.int64)})
        return np.asarray(result[self._compiled.output("logits")])[:rows * self._factor]
