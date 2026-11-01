"""The native CPU decoder (native/tdt) against NumPy doing the same int8 arithmetic.

Needs the library built (``python native/tdt/build.py``; CI builds it) and is skipped where it isn't.
Small random weights, as the other decoder tests use. The integer products are exact and the float steps
are done in the same order, so the tokens must match one for one; NumPy's walk over the frames is the
reference for the native one.
"""

import numpy as np
import pytest
import torch

from skryb.transcription.native_engine_dll import DEV_BUILD_TDT_DLL
from skryb.transcription.openvino_int8_weights import int8_rows
from skryb.transcription.parakeet_numpy import greedy_tdt_walk
from tests.transcription.test_parakeet_numpy import _random_decoding_parts

pytestmark = pytest.mark.skipif(not DEV_BUILD_TDT_DLL.is_file(), reason="native/tdt is not built")

from skryb.transcription.native_tdt_decoder import (  # noqa: E402
    KERNEL_AVX2,
    KERNEL_SCALAR,
    KERNEL_VNNI_INT8,
    NativeTdtDecoder,
)

_F = np.float32
_HIDDEN = 640
_TOKENS = 8193


def _quantize(x: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """As tdt_cpu rounds a product's input: one scale per vector, half to even."""
    largest = _F(np.abs(x).max())
    if largest == 0:
        return np.zeros(x.shape, np.int32), _F(0)
    return np.rint(x * (_F(127) / largest)).astype(np.int32), largest / _F(127)


class _Int8Reference:
    """tdt_cpu's arithmetic in NumPy: int8 weight rows, int8 inputs, int32 sums, float32 for the rest."""

    def __init__(self, tensors: dict[str, np.ndarray]) -> None:
        self._embed = tensors["decoder.embed.weight"].astype(_F)
        self._lstm = []
        for layer in (0, 1):
            weight = np.concatenate(
                [tensors[f"decoder.lstm.weight_ih_l{layer}"], tensors[f"decoder.lstm.weight_hh_l{layer}"]], axis=1,
            )
            q, scale = int8_rows(weight.astype(_F))
            bias = (tensors[f"decoder.lstm.bias_ih_l{layer}"] + tensors[f"decoder.lstm.bias_hh_l{layer}"]).astype(_F)
            self._lstm.append((q.astype(np.int32), scale.reshape(-1), bias))
        self._pred = self._rows(tensors, "joint.pred")
        self._out = self._rows(tensors, "joint.joint_net.2")
        self._enc = (tensors["joint.enc.weight"].astype(_F).T, tensors["joint.enc.bias"].astype(_F))

    @staticmethod
    def _rows(tensors: dict[str, np.ndarray], name: str) -> tuple:
        q, scale = int8_rows(tensors[f"{name}.weight"].astype(_F))
        return q.astype(np.int32), scale.reshape(-1), tensors[f"{name}.bias"].astype(_F)

    def decode(self, encoded: np.ndarray) -> list[int]:
        enc = encoded.astype(_F) @ self._enc[0] + self._enc[1]
        zeros = np.zeros(_HIDDEN, _F)
        return greedy_tdt_walk(enc, self._step, self._score, [zeros] * 4)

    def _step(self, label: int, state: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
        x, new_state = self._embed[label], []
        for layer, (q, scale, bias) in enumerate(self._lstm):
            h, c = state[2 * layer], state[2 * layer + 1]
            xq, sx = _quantize(x)
            hq, sh = _quantize(h)
            gates = (q[:, :_HIDDEN] @ xq).astype(_F) * (sx * scale) + (q[:, _HIDDEN:] @ hq).astype(_F) * (sh * scale) + bias
            i, f, g, o = gates.reshape(4, _HIDDEN)
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = _sigmoid(o) * np.tanh(c)
            new_state += [h, c]
            x = h
        q, scale, bias = self._pred
        hq, s = _quantize(x)
        return (q @ hq).astype(_F) * (s * scale) + bias, new_state

    def _score(self, enc: np.ndarray, pred: np.ndarray) -> list[tuple[int, int]]:
        q, scale, bias = self._out
        best = []
        for frame in enc:
            joined, s = _quantize(np.maximum(frame + pred, _F(0)))
            logits = (q @ joined).astype(_F) * (s * scale) + bias
            best.append((int(logits[:_TOKENS].argmax()), int(logits[_TOKENS:].argmax())))
        return best


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return _F(1) / (_F(1) + np.exp(-x))


def _parts(seed: int, frames: int = 150) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    encoded = torch.randn(frames, 1024, generator=torch.Generator().manual_seed(seed))
    _, _, tensors = _random_decoding_parts(seed, encoded)
    return encoded.numpy(), tensors


def _kernels_this_cpu_has(tensors: dict[str, np.ndarray]) -> list[int]:
    kernels = []
    for kernel in (KERNEL_SCALAR, KERNEL_AVX2, KERNEL_VNNI_INT8):
        try:
            NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors, kernel=kernel, threads=1)
        except OSError:
            continue
        kernels.append(kernel)
    return kernels


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_native_decoder_gives_numpys_int8_tokens_one_for_one(seed: int) -> None:
    encoded, tensors = _parts(seed)

    expected = _Int8Reference(tensors).decode(encoded)
    tokens = NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors).decode(encoded)

    assert len(expected) > 20
    assert tokens == expected


def test_every_kernel_this_cpu_has_gives_the_same_tokens() -> None:
    encoded, tensors = _parts(4)

    results = {kernel: NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors, kernel=kernel).decode(encoded)
               for kernel in _kernels_this_cpu_has(tensors)}

    assert KERNEL_SCALAR in results
    assert len({tuple(tokens) for tokens in results.values()}) == 1


@pytest.mark.parametrize("threads", [1, 2, 3, 7, 16])
def test_sharing_the_work_among_threads_changes_no_token(threads: int) -> None:
    encoded, tensors = _parts(5)

    alone = NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors, threads=1).decode(encoded)
    shared = NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors, threads=threads).decode(encoded)

    assert shared == alone


def test_one_decoder_gives_the_same_tokens_every_time_over_a_hundred_clips() -> None:
    encoded, tensors = _parts(6, frames=60)
    decoder = NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors)

    first = decoder.decode(encoded)

    assert all(decoder.decode(encoded) == first for _ in range(100))


def test_ten_minutes_of_frames_decode_without_running_out_of_room() -> None:
    encoded, tensors = _parts(7, frames=7500)

    tokens = NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors).decode(encoded)

    assert len(tokens) > 100
    assert all(0 <= token < _TOKENS - 1 for token in tokens)


def test_an_empty_encoding_and_a_single_frame_decode_cleanly() -> None:
    encoded, tensors = _parts(8, frames=1)
    decoder = NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors)

    assert decoder.decode(np.zeros((0, 1024), np.float32)) == []
    assert decoder.decode(encoded) == _Int8Reference(tensors).decode(encoded)


def test_an_impossible_thread_count_is_refused() -> None:
    _, tensors = _parts(9, frames=1)

    with pytest.raises(OSError, match="refused"):
        NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors, threads=0)


def test_the_npu_engine_decodes_natively_when_the_library_is_there_and_through_openvino_when_not(
    tmp_path, monkeypatch,
) -> None:
    openvino = pytest.importorskip("openvino")
    from skryb.transcription import openvino_parakeet_adapter as adapter
    from skryb.transcription.openvino_tdt_decoder import OpenVinoTdtDecoder

    _, tensors = _parts(10, frames=1)
    np.savez(tmp_path / "decoder.npz", **tensors)

    monkeypatch.setenv("TDT_CPU_DLL", str(DEV_BUILD_TDT_DLL))
    assert isinstance(adapter._load_decoder(openvino, tmp_path / "decoder.npz"), NativeTdtDecoder)
    monkeypatch.setenv("TDT_CPU_DLL", str(tmp_path / "missing.dll"))
    assert isinstance(adapter._load_decoder(openvino, tmp_path / "decoder.npz"), OpenVinoTdtDecoder)


def _speechlike(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = int(seconds * 16000)
    envelope = 0.5 + 0.5 * np.sin(np.linspace(0, seconds * 6, count))
    return (rng.normal(0, 0.1, count) * envelope).astype(np.float32)


@pytest.mark.parametrize("seconds", [257 / 16000, 0.5, 3.0, 10.23, 30.7])
def test_the_native_features_match_numpys_to_float_rounding(seconds: float) -> None:
    from skryb.transcription.parakeet_numpy import log_mel_features

    _, tensors = _parts(11, frames=1)
    samples = _speechlike(seconds, seed=int(seconds * 100))

    native = NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors).log_mel(samples)
    expected = log_mel_features(samples)

    assert native.shape == expected.shape
    assert np.abs(native - expected).max() < 1e-3


def test_every_kernel_and_thread_count_gives_the_same_features_bit_for_bit() -> None:
    _, tensors = _parts(12, frames=1)
    samples = _speechlike(7.3, seed=3)

    results = [
        NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors, kernel=kernel, threads=threads).log_mel(samples)
        for kernel in _kernels_this_cpu_has(tensors) for threads in (1, 3)
    ]

    assert all(np.array_equal(features, results[0]) for features in results)


def test_audio_too_short_for_a_frame_on_each_side_gets_numpys_features() -> None:
    from skryb.transcription.parakeet_numpy import log_mel_features

    _, tensors = _parts(13, frames=1)
    samples = _speechlike(256 / 16000, seed=4)

    assert np.array_equal(NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors).log_mel(samples), log_mel_features(samples))


def test_silence_and_ten_minutes_of_speech_give_finite_features_of_the_right_shape() -> None:
    _, tensors = _parts(14, frames=1)
    decoder = NativeTdtDecoder(DEV_BUILD_TDT_DLL, tensors)

    silence = decoder.log_mel(np.zeros(16000, np.float32))
    long = decoder.log_mel(_speechlike(600, seed=5))

    assert silence.shape == (1, 128, 101) and np.isfinite(silence).all()
    assert long.shape == (1, 128, 60001) and np.isfinite(long).all()
