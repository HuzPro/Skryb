"""Tests for the OpenVINO (NPU) Parakeet engine.

Everything here runs on the CPU with small randomly initialised modules
and fakes for OpenVINO, so it passes the same on any machine. Real
weights on a real NPU stay a manual check (scripts/check_npu.py).
"""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import torch

from skryb.core.exceptions import TranscriptionError
from skryb.transcription.openvino_parakeet_adapter import (
    CompiledEncoder,
    bucket_frames,
    compile_encoder,
    split_at_pauses,
)
from skryb.transcription.openvino_parakeet_export import MaskedEncoder


_RATE = 16_000
_LONGEST = 30 * _RATE


def _small_encoder() -> torch.nn.Module:
    from nano_parakeet.model import FastConformerEncoder

    torch.manual_seed(7)
    encoder = FastConformerEncoder(d_model=64, n_layers=2, n_heads=4, d_ff=128)
    for layer in encoder.layers:
        layer.conv.batch_norm.running_mean.uniform_(-0.5, 0.5)
        layer.conv.batch_norm.running_var.uniform_(0.5, 2.0)
    return encoder.eval()


class TestPaddedClipsEncodeLikeUnpaddedOnes:
    @pytest.mark.parametrize("length", [9, 200, 201, 203, 208, 401])
    def test_the_real_frames_match_the_unpadded_clip_whatever_the_padding(self, length: int) -> None:
        encoder = _small_encoder()
        features = torch.randn(1, 128, length)
        padded = torch.zeros(1, 128, 401)
        padded[..., :length] = features

        with torch.inference_mode():
            expected, expected_length = encoder(features, torch.tensor([length]))
            encoded, encoded_length = MaskedEncoder(encoder)(padded, torch.tensor([length]))

        assert int(encoded_length) == int(expected_length) == expected.shape[1]
        torch.testing.assert_close(encoded[:, :expected.shape[1]], expected, atol=1e-5, rtol=1e-4)

    def test_without_the_masking_the_padding_leaks_into_the_last_frames(self) -> None:
        encoder = _small_encoder()
        features = torch.randn(1, 128, 203)
        padded = torch.zeros(1, 128, 401)
        padded[..., :203] = features

        with torch.inference_mode():
            expected, length = encoder(features, torch.tensor([203]))
            leaked, _ = encoder(padded, torch.tensor([401]))

        assert not torch.allclose(leaked[:, int(length) - 1], expected[:, -1], atol=1e-3)


class TestSplittingLongAudio:
    def test_audio_up_to_the_limit_stays_whole(self) -> None:
        samples = np.random.default_rng(1).normal(0, 0.1, _LONGEST).astype(np.float32)

        assert [len(p) for p in split_at_pauses(samples, _LONGEST)] == [_LONGEST]

    def test_one_sample_over_the_limit_makes_two_pieces(self) -> None:
        samples = np.random.default_rng(2).normal(0, 0.1, _LONGEST + 1).astype(np.float32)

        pieces = split_at_pauses(samples, _LONGEST)

        assert len(pieces) == 2
        assert np.array_equal(np.concatenate(pieces), samples)

    def test_long_dictation_is_cut_in_its_pauses(self) -> None:
        samples = np.random.default_rng(3).normal(0, 0.1, 70 * _RATE).astype(np.float32)
        samples[27 * _RATE:int(27.3 * _RATE)] = 0.0
        samples[54 * _RATE:int(54.3 * _RATE)] = 0.0

        pieces = split_at_pauses(samples, _LONGEST)
        cuts = np.cumsum([len(p) for p in pieces])[:-1] / _RATE

        assert len(pieces) == 3
        assert 27.0 <= cuts[0] <= 27.3
        assert 54.0 <= cuts[1] <= 54.3
        assert np.array_equal(np.concatenate(pieces), samples)

    def test_seven_minutes_without_a_pause_is_cut_at_the_limit_and_loses_nothing(self) -> None:
        samples = np.random.default_rng(4).normal(0, 0.1, 408 * _RATE).astype(np.float32)

        pieces = split_at_pauses(samples, _LONGEST)

        assert all(0 < len(p) <= _LONGEST for p in pieces)
        assert len(pieces) >= 14
        assert np.array_equal(np.concatenate(pieces), samples)

    def test_empty_audio_has_no_pieces(self) -> None:
        assert split_at_pauses(np.zeros(0, np.float32), _LONGEST) == []


class _FakeCompiled:
    def __init__(self, frames_out_per_in: int = 8, value: float = 1.0) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self._ratio = frames_out_per_in
        self._value = value

    def __call__(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        features, length = inputs
        self.calls.append((features.shape, int(length[0])))
        assert not features[..., int(length[0]):].any(), "padding must be zeros"
        frames = features.shape[-1] // self._ratio + 1
        real = (int(length[0]) - 1) // self._ratio + 1
        return [np.full((1, frames, 4), self._value, np.float32), np.array([real], np.int64)]


class TestFixedLengths:
    def test_features_are_padded_to_the_shortest_length_that_fits(self) -> None:
        short, long = _FakeCompiled(), _FakeCompiled()
        encoder = CompiledEncoder("NPU", {bucket_frames(10): short, bucket_frames(30): long})

        encoded = encoder(np.ones((1, 128, 541), np.float32))

        assert short.calls == [((1, 128, 1001), 541)]
        assert long.calls == []
        assert encoded.shape == (68, 4)

    def test_a_clip_exactly_at_a_length_is_not_padded_further(self) -> None:
        short, long = _FakeCompiled(), _FakeCompiled()
        encoder = CompiledEncoder("NPU", {bucket_frames(10): short, bucket_frames(30): long})

        encoder(np.ones((1, 128, 1001), np.float32))

        assert short.calls == [((1, 128, 1001), 1001)]

    def test_a_clip_longer_than_every_length_is_refused(self) -> None:
        encoder = CompiledEncoder("NPU", {bucket_frames(10): _FakeCompiled()})

        with pytest.raises(TranscriptionError):
            encoder(np.ones((1, 128, 1002), np.float32))

    def test_a_device_that_takes_any_length_gets_the_clip_unpadded(self) -> None:
        any_length = _FakeCompiled()

        CompiledEncoder("CPU", {0: any_length})(np.ones((1, 128, 77), np.float32))

        assert any_length.calls == [((1, 128, 77), 77)]


class _FakeModel:
    def __init__(self) -> None:
        self.shapes: dict | None = None

    def reshape(self, shapes: dict) -> None:
        self.shapes = shapes


class _FakeCore:
    def __init__(self, behaviour: dict[str, str]) -> None:
        self._behaviour = behaviour
        self.compiled: list[tuple[str, dict | None]] = []

    def read_model(self, path: str) -> _FakeModel:
        return _FakeModel()

    def compile_model(self, model: _FakeModel, device: str, config: dict) -> _FakeCompiled:
        assert config["CACHE_DIR"].endswith("compiled")
        self.compiled.append((device, model.shapes))
        if self._behaviour[device] == "refuse":
            raise RuntimeError(f"{device} cannot compile this graph")
        return _FakeCompiled(value=np.nan if self._behaviour[device] == "nan" else 1.0)


class _FakeOpenVino:
    def __init__(self, behaviour: dict[str, str]) -> None:
        self.core = _FakeCore(behaviour)

    def Core(self) -> _FakeCore:  # noqa: N802 (OpenVINO's name)
        return self.core


class TestChoosingADevice:
    def test_the_npu_gets_one_compile_per_fixed_length(self, tmp_path) -> None:
        openvino = _FakeOpenVino({"NPU": "ok", "CPU": "ok"})

        encoder = compile_encoder(openvino, tmp_path / "encoder.xml", ["NPU", "CPU"])

        assert encoder.device == "NPU"
        assert [shapes["features"] for _, shapes in openvino.core.compiled] == [
            [1, 128, 1001], [1, 128, 3001],
        ]

    def test_an_npu_that_refuses_the_graph_falls_back_to_the_cpu(self, tmp_path) -> None:
        openvino = _FakeOpenVino({"NPU": "refuse", "CPU": "ok"})

        encoder = compile_encoder(openvino, tmp_path / "encoder.xml", ["NPU", "CPU"])

        assert encoder.device == "CPU"
        assert openvino.core.compiled[-1] == ("CPU", None)

    def test_an_npu_that_overflows_to_nan_falls_back_to_the_cpu(self, tmp_path) -> None:
        openvino = _FakeOpenVino({"NPU": "nan", "CPU": "ok"})

        assert compile_encoder(openvino, tmp_path / "encoder.xml", ["NPU", "CPU"]).device == "CPU"

    def test_no_working_device_is_an_error_naming_what_was_tried(self, tmp_path) -> None:
        openvino = _FakeOpenVino({"NPU": "refuse", "CPU": "refuse"})

        with pytest.raises(TranscriptionError, match="NPU, CPU"):
            compile_encoder(openvino, tmp_path / "encoder.xml", ["NPU", "CPU"])


@pytest.mark.skipif(importlib.util.find_spec("openvino") is None, reason="openvino is not installed")
def test_openvinos_usage_reporting_is_replaced_by_its_no_op_stub() -> None:
    code = (
        "from skryb.transcription.openvino_parakeet_adapter import import_openvino\n"
        "import_openvino()\n"
        "import openvino.tools.ovc.telemetry_utils as telemetry\n"
        "print(telemetry.tm.__name__)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True,
    )

    assert completed.stdout.strip().endswith("telemetry_stub")
