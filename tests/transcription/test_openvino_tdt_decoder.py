"""The OpenVINO CPU decoder against the NumPy one it speeds up.

Runs on OpenVINO's CPU device, which every machine has, with small random
weights. The int8 weights are compared through the NumPy decoder given the
same weights expanded back to fp32, so the check is the graph, not rounding.
"""

import numpy as np
import pytest

from skryb.transcription.parakeet_numpy import TdtDecoder
from tests.transcription.test_parakeet_numpy import _random_decoding_parts

openvino = pytest.importorskip("openvino")

from skryb.transcription.openvino_int8_weights import int8_rows  # noqa: E402
from skryb.transcription.openvino_tdt_decoder import OpenVinoTdtDecoder  # noqa: E402


def _as_the_cpu_sees_them(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """*tensors* with every int8-stored weight rounded the way the graph stores it."""
    rounded = dict(tensors)
    for layer in (0, 1):
        ih, hh = f"decoder.lstm.weight_ih_l{layer}", f"decoder.lstm.weight_hh_l{layer}"
        q, scale = int8_rows(np.concatenate([tensors[ih], tensors[hh]], axis=1))
        expanded = q.astype(np.float32) * scale
        rounded[ih], rounded[hh] = expanded[:, :tensors[ih].shape[1]], expanded[:, tensors[ih].shape[1]:]
    for name in ("joint.pred.weight", "joint.joint_net.2.weight"):
        q, scale = int8_rows(tensors[name])
        rounded[name] = q.astype(np.float32) * scale
    return rounded


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_cpu_graphs_decode_the_same_tokens_as_numpy_given_the_same_weights(seed: int) -> None:
    import torch

    encoded = torch.randn(150, 1024, generator=torch.Generator().manual_seed(seed))
    _, _, tensors = _random_decoding_parts(seed, encoded)

    expected = TdtDecoder(_as_the_cpu_sees_them(tensors)).decode(encoded.numpy())
    tokens = OpenVinoTdtDecoder(openvino, tensors, exact=True).decode(encoded.numpy())

    assert len(expected) > 20
    assert tokens == expected


def test_an_empty_encoding_decodes_to_nothing() -> None:
    _, _, tensors = _random_decoding_parts(0)

    assert OpenVinoTdtDecoder(openvino, tensors).decode(np.zeros((0, 1024), np.float32)) == []


def test_the_exact_decoder_turns_activation_rounding_off_and_the_default_keeps_the_plugins_choice() -> None:
    _, _, tensors = _random_decoding_parts(0)
    key = "DYNAMIC_QUANTIZATION_GROUP_SIZE"

    default = OpenVinoTdtDecoder(openvino, tensors)._score_request.get_compiled_model()
    exact = OpenVinoTdtDecoder(openvino, tensors, exact=True)._score_request.get_compiled_model()

    assert exact.get_property(key) == 0
    assert default.get_property(key) == openvino.Core().get_property("CPU", key)
