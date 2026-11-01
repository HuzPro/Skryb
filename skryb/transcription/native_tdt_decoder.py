"""Greedy TDT decoding through ``tdt_cpu`` (``native/tdt``): the walk, the LSTM and the joint in one native call.

The library also makes the log-mel features in front of the encoder: the FFT on eight frames at a time in
AVX2 lanes, the mel filters' nonzero weights only, the frames split among the same threads.

The same int8 weight rows as ``openvino_tdt_decoder``, with each product's input rounded to int8 per vector,
run as int8 dot products (AVX-VNNI-INT8, or AVX2). OpenVINO spent 140-290 us on each of the two small
products per token on a Core 5 320; here the whole clip is one call.
"""

import ctypes
import sys
from pathlib import Path

import numpy as np

from skryb.transcription.openvino_int8_weights import int8_rows
from skryb.transcription.parakeet_numpy import _filterbank, _window, log_mel_features

KERNEL_AUTO, KERNEL_SCALAR, KERNEL_AVX2, KERNEL_VNNI_INT8 = 0, 1, 2, 3
KERNEL_NAMES = {KERNEL_SCALAR: "C", KERNEL_AVX2: "AVX2", KERNEL_VNNI_INT8: "AVX-VNNI-INT8"}
_MAX_SYMBOLS_PER_FRAME = 10
_HOP = 160
_MELS = 128
_ARRAY = np.ctypeslib.ndpointer(flags="C_CONTIGUOUS")


class NativeTdtDecoder:
    """``parakeet_numpy.TdtDecoder``'s decoding with int8 products, in one native call per clip."""

    def __init__(
        self, library_path: Path, tensors: dict[str, np.ndarray], kernel: int = KERNEL_AUTO, threads: int = 3,
    ) -> None:
        self._library = _load(library_path)
        self._enc = (
            np.ascontiguousarray(tensors["joint.enc.weight"], np.float32).T.copy(),
            np.asarray(tensors["joint.enc.bias"], np.float32),
        )
        # Kept alive beside the handle: the library copies them, but only once tdt_create returns.
        arrays = [np.ascontiguousarray(tensors["decoder.embed.weight"], np.float32)]
        for layer in (0, 1):
            weight = np.concatenate(
                [tensors[f"decoder.lstm.weight_ih_l{layer}"], tensors[f"decoder.lstm.weight_hh_l{layer}"]], axis=1,
            )
            bias = tensors[f"decoder.lstm.bias_ih_l{layer}"] + tensors[f"decoder.lstm.bias_hh_l{layer}"]
            arrays += _rows(weight, bias)
        arrays += _rows(tensors["joint.pred.weight"], tensors["joint.pred.bias"])
        arrays += _rows(tensors["joint.joint_net.2.weight"], tensors["joint.joint_net.2.bias"])
        self._handle = self._library.tdt_create(kernel, threads, *arrays)
        if not self._handle:
            raise OSError(f"tdt_cpu refused kernel {kernel}: this CPU lacks it, or memory ran out")
        if not self._library.tdt_set_mel(self._handle, _window(), np.ascontiguousarray(_filterbank())):
            raise OSError("tdt_cpu ran out of memory for the mel filters")

    @property
    def kernel(self) -> str:
        """Which dot product runs: "AVX-VNNI-INT8", "AVX2" or "C"."""
        return KERNEL_NAMES[self._library.tdt_kernel(self._handle)]

    def log_mel(self, samples: np.ndarray) -> np.ndarray:
        """``parakeet_numpy.log_mel_features``, natively: ``[1, 128, frames]``."""
        samples = np.ascontiguousarray(samples, np.float32)
        frames = 1 + len(samples) // _HOP
        features = np.empty((1, _MELS, frames), np.float32)
        if self._library.tdt_log_mel(self._handle, samples, len(samples), features) != frames:
            return log_mel_features(samples)
        return features

    def decode(self, encoded: np.ndarray) -> list[int]:
        """Token ids for ``[frames, 1024]`` encoder output."""
        enc = np.ascontiguousarray(encoded.astype(np.float32) @ self._enc[0] + self._enc[1])
        tokens = np.empty(max(1, len(enc) * _MAX_SYMBOLS_PER_FRAME), np.int32)
        count = self._library.tdt_decode(self._handle, enc, len(enc), tokens, len(tokens))
        if count < 0:
            raise RuntimeError("tdt_cpu ran out of room for tokens")
        return tokens[:count].tolist()

    def __del__(self) -> None:
        handle, self._handle = getattr(self, "_handle", None), None
        if handle:
            self._library.tdt_destroy(handle)


def library_name() -> str:
    return "tdt_cpu.dll" if sys.platform == "win32" else "libtdt_cpu.so"


def _rows(weight: np.ndarray, bias: np.ndarray) -> list[np.ndarray]:
    """*weight* as int8 rows with one scale each, and the bias, laid out for tdt_create."""
    q, scale = int8_rows(np.asarray(weight, np.float32))
    return [np.ascontiguousarray(q), np.ascontiguousarray(scale.reshape(-1)), np.ascontiguousarray(bias, np.float32)]


def _load(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.tdt_create.restype = ctypes.c_void_p
    library.tdt_create.argtypes = [ctypes.c_int, ctypes.c_int] + [_ARRAY] * 13
    library.tdt_kernel.restype = ctypes.c_int
    library.tdt_kernel.argtypes = [ctypes.c_void_p]
    library.tdt_decode.restype = ctypes.c_int
    library.tdt_decode.argtypes = [ctypes.c_void_p, _ARRAY, ctypes.c_int, _ARRAY, ctypes.c_int]
    library.tdt_set_mel.restype = ctypes.c_int
    library.tdt_set_mel.argtypes = [ctypes.c_void_p, _ARRAY, _ARRAY]
    library.tdt_log_mel.restype = ctypes.c_int
    library.tdt_log_mel.argtypes = [ctypes.c_void_p, _ARRAY, ctypes.c_int, _ARRAY]
    library.tdt_destroy.restype = None
    library.tdt_destroy.argtypes = [ctypes.c_void_p]
    return library
