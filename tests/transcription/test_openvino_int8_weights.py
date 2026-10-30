"""Rewriting an OpenVINO graph's weights as int8, on OpenVINO's CPU device with small random graphs."""

import numpy as np
import pytest

from skryb.transcription.openvino_int8_weights import compress_encoder, int8_rows, opset

openvino = pytest.importorskip("openvino")
ops = opset(openvino)


def _graph(path, rng) -> tuple[np.ndarray, np.ndarray]:
    """A conv, a MatMul against [N, K] and one against [K, N], saved the way the export saves (fp16)."""
    x = ops.parameter([1, 16, 50], np.float32, name="features")
    conv_w = rng.standard_normal((96, 16, 3)).astype(np.float32) * 0.2
    y = ops.convolution(x, ops.constant(conv_w), [1], [1], [1], [1])
    y = ops.transpose(y, ops.constant(np.array([0, 2, 1], np.int64)))
    nk = rng.standard_normal((128, 96)).astype(np.float32) * np.logspace(-2, 0, 128)[:, None].astype(np.float32)
    y = ops.matmul(y, ops.constant(nk), False, True)
    kn = rng.standard_normal((128, 64)).astype(np.float32) * 0.1
    y = ops.matmul(ops.relu(y), ops.constant(kn), False, False)
    small = rng.standard_normal((64, 8)).astype(np.float32)
    y = ops.matmul(y, ops.constant(small), False, False)
    openvino.save_model(openvino.Model([y], [x], "tiny"), str(path), compress_to_fp16=True)
    return rng.standard_normal((1, 16, 50)).astype(np.float32), small


def _run(path, features) -> np.ndarray:
    return openvino.Core().compile_model(str(path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})([features])[0]


def _constant_types(path) -> list[str]:
    model = openvino.Core().read_model(str(path))
    return [
        node.get_element_type().get_type_name()
        for node in model.get_ordered_ops() if node.get_type_name() == "Constant"
    ]


def test_the_int8_graph_gives_nearly_the_same_output_as_the_fp16_one(tmp_path) -> None:
    source, target = tmp_path / "encoder.xml", tmp_path / "encoder-int8.xml"
    features, _ = _graph(source, np.random.default_rng(0))

    changed = compress_encoder(openvino, source, target)

    expected, got = _run(source, features), _run(target, features)
    assert changed == 3
    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 0.01


def test_only_the_large_weights_become_int8_and_the_small_one_is_left_alone(tmp_path) -> None:
    source, target = tmp_path / "encoder.xml", tmp_path / "encoder-int8.xml"
    _graph(source, np.random.default_rng(1))

    compress_encoder(openvino, source, target)

    types = _constant_types(target)
    assert types.count("i8") == 3
    assert types.count("f16") == 1


def test_the_rewrite_leaves_no_staging_files_and_keeps_the_source(tmp_path) -> None:
    source, target = tmp_path / "encoder.xml", tmp_path / "encoder-int8.xml"
    _graph(source, np.random.default_rng(2))

    compress_encoder(openvino, source, target)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "encoder-int8.bin", "encoder-int8.xml", "encoder.bin", "encoder.xml",
    ]


def test_int8_rows_are_within_half_a_step_of_the_weights_for_any_rank() -> None:
    rng = np.random.default_rng(3)
    for shape in [(300, 70), (96, 16, 3), (8, 4, 3, 3)]:
        weight = rng.standard_normal(shape).astype(np.float32) * 3

        q, scale = int8_rows(weight)

        assert q.dtype == np.int8 and q.shape == shape
        assert scale.shape == (shape[0],) + (1,) * (len(shape) - 1)
        assert np.all(np.abs(q.astype(np.float32) * scale - weight) <= scale / 2 + 1e-6)


def test_an_all_zero_row_stays_zero_instead_of_dividing_by_zero() -> None:
    weight = np.zeros((3, 8), np.float32)
    weight[1] = 0.5

    q, scale = int8_rows(weight)

    assert np.isfinite(scale).all()
    assert not q[0].any() and not q[2].any()
