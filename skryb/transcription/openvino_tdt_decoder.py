"""Greedy TDT decoding with the prediction net and joint on OpenVINO's CPU device, weights in int8.

Decoding is one small step per token, so it is bound by reading weights: the LSTM, the prediction
projection and the joint's output layer are about 34 MB at fp32, read once per token. OpenVINO's CPU
plugin keeps symmetric per-channel int8 weights compressed and expands them as it multiplies, which
reads a quarter of that. On a Core 5 320 this decodes 3 s of dictation in about 20 ms where NumPy took 70.
The walk over frames is ``parakeet_numpy``'s; only the two products it calls per token run here.

The plugin also rounds the activations to int8 on the fly. On random weights that drifts from
NumPy's tokens after a few dozen, so the tests run with it off (``exact``); on 100 real dictations
(4,896 tokens) it changed none, and it decodes 1.24x faster.
"""

import numpy as np

from skryb.transcription.openvino_int8_weights import int8_rows, opset
from skryb.transcription.parakeet_numpy import greedy_tdt_walk

_HIDDEN = 640
_TOKENS_WITH_BLANK = 8193
_DURATION_COUNT = 5
# Two threads decode fastest on a Core 5 320 (two P-cores): one is 15% slower, four no faster.
_CPU_CONFIG = {
    "PERFORMANCE_HINT": "LATENCY",
    "INFERENCE_NUM_THREADS": 2,
    # The CPU may offer bf16 or fp16 maths; the NumPy decoder these tokens must match is fp32.
    "INFERENCE_PRECISION_HINT": "f32",
}
_EXACT_CONFIG = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}
_STATE_NAMES = ("h0", "c0", "h1", "c1")


class OpenVinoTdtDecoder:
    """``parakeet_numpy.TdtDecoder``'s decoding, with each token's products compiled for the CPU."""

    def __init__(self, openvino: object, tensors: dict[str, np.ndarray], exact: bool = False) -> None:
        ops = opset(openvino)
        config = {**_CPU_CONFIG, **(_EXACT_CONFIG if exact else {})}
        self._embed = np.ascontiguousarray(tensors["decoder.embed.weight"], np.float32)
        self._enc = (
            np.ascontiguousarray(tensors["joint.enc.weight"], np.float32).T.copy(),
            np.asarray(tensors["joint.enc.bias"], np.float32),
        )
        core = openvino.Core()
        self._step_request = core.compile_model(
            _step_model(openvino, ops, tensors), "CPU", config,
        ).create_infer_request()
        self._score_request = core.compile_model(
            _score_model(openvino, ops, tensors), "CPU", config,
        ).create_infer_request()

    def decode(self, encoded: np.ndarray) -> list[int]:
        """Token ids for ``[frames, 1024]`` encoder output."""
        enc = encoded.astype(np.float32) @ self._enc[0] + self._enc[1]
        zeros = np.zeros((1, _HIDDEN), np.float32)
        return greedy_tdt_walk(enc, self._step, self._score, [zeros] * len(_STATE_NAMES))

    def _step(self, label: int, state: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        request = self._step_request
        request.infer([self._embed[label][None], *state])
        pred, *new_state = (request.get_output_tensor(i).data.copy() for i in range(1 + len(_STATE_NAMES)))
        return pred[0], new_state

    def _score(self, enc: np.ndarray, pred: np.ndarray) -> list[tuple[int, int]]:
        request = self._score_request
        request.infer([np.ascontiguousarray(enc), pred[None]])
        tokens = request.get_output_tensor(0).data[:, 0].tolist()
        durations = request.get_output_tensor(1).data[:, 0].tolist()
        return list(zip(tokens, durations))


def _int8_linear(ops: object, x: object, weight: np.ndarray, bias: np.ndarray) -> object:
    """``x @ weight.T + bias``, with *weight* stored as int8 and expanded by the plugin."""
    q, scale = int8_rows(np.asarray(weight, np.float32))
    expanded = ops.multiply(ops.convert(ops.constant(q), "f32"), ops.constant(scale))
    return ops.add(ops.matmul(x, expanded, False, True), ops.constant(np.asarray(bias, np.float32)[None]))


def _step_model(openvino: object, ops: object, w: dict[str, np.ndarray]) -> object:
    """The two-layer LSTM and the joint's prediction projection, for one label embedding."""
    x = ops.parameter([1, _HIDDEN], np.float32, name="x")
    states = [ops.parameter([1, _HIDDEN], np.float32, name=name) for name in _STATE_NAMES]
    outputs, inp = [], x
    for layer in (0, 1):
        h, c = states[2 * layer], states[2 * layer + 1]
        weight = np.concatenate([w[f"decoder.lstm.weight_ih_l{layer}"], w[f"decoder.lstm.weight_hh_l{layer}"]], axis=1)
        bias = w[f"decoder.lstm.bias_ih_l{layer}"] + w[f"decoder.lstm.bias_hh_l{layer}"]
        gates = _int8_linear(ops, ops.concat([inp, h], 1), weight, bias)
        i, f, g, o = ops.split(gates, 1, 4).outputs()
        c = ops.add(ops.multiply(ops.sigmoid(f), c), ops.multiply(ops.sigmoid(i), ops.tanh(g)))
        h = ops.multiply(ops.sigmoid(o), ops.tanh(c))
        outputs += [h, c]
        inp = h
    pred = _int8_linear(ops, inp, w["joint.pred.weight"], w["joint.pred.bias"])
    return openvino.Model([pred, *outputs], [x, *states], "tdt_step")


def _score_model(openvino: object, ops: object, w: dict[str, np.ndarray]) -> object:
    """The joint for a window of projected frames against one prediction: best token and duration per frame."""
    enc = ops.parameter([-1, _HIDDEN], np.float32, name="enc")
    pred = ops.parameter([1, _HIDDEN], np.float32, name="pred")
    logits = _int8_linear(ops, ops.relu(ops.add(enc, pred)), w["joint.joint_net.2.weight"], w["joint.joint_net.2.bias"])
    tokens, durations = ops.variadic_split(logits, 1, [_TOKENS_WITH_BLANK, _DURATION_COUNT]).outputs()
    best_token = ops.topk(tokens, 1, 1, "max", "none", "i32").output(1)
    best_duration = ops.topk(durations, 1, 1, "max", "none", "i32").output(1)
    return openvino.Model([best_token, best_duration], [enc, pred], "tdt_score")
