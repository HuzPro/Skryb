"""Parakeet TDT 0.6B (Parakeet Ultra, Underdog Bark) on the native HIP engine (``native/parakeet``), no torch.

The engine is a DLL of hand-written kernels for the RX 9060 XT (gfx1200). It reads one packed file laid
out as it sits in VRAM, built once from the downloaded snapshot inside it (``snapshots/<sha>/.npk``), the
way the native Qwen3-ASR engine packs ``.qasr``.
"""

import ctypes
import importlib.util
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.native_engine_dll import native_parakeet_dll
from skryb.transcription.parakeet_loader import transformers_layout_weights
from skryb.transcription.result import TranscriptionResult

_logger = logging.getLogger(__name__)

_PACK_DIRECTORY_NAME = ".npk"
_PACK_FILENAME = "parakeet.bin"
_SAMPLE_RATE = 16000
_TEXT_BUFFER_BYTES = 1 << 20
_ERROR_BUFFER_BYTES = 1024
# nano-parakeet transcribes in 60 s pieces; the same cut keeps the two engines' text the same.
_CHUNK_SECONDS = 60


def packed_model_path(weights: Path) -> Path:
    """Where the packed engine file for the snapshot holding *weights* lives."""
    snapshot = weights.parent.parent if weights.parent.name == ".bark" else weights.parent
    return snapshot / _PACK_DIRECTORY_NAME / _PACK_FILENAME


class NativeParakeetAdapter(TranscriptionEngine):
    """Batch Parakeet TDT through ``npk.dll``. One call at a time, like every adapter here."""

    def __init__(self, model_dir: Path, repo_id: str, dll_path: Path | None = None) -> None:
        self._model_dir = model_dir
        self._repo_id = repo_id
        self._dll_path = dll_path or native_parakeet_dll() or Path("npk.dll")
        self._lib: ctypes.CDLL | None = None
        self._handle: int | None = None
        self._text = ctypes.create_string_buffer(_TEXT_BUFFER_BYTES)

    def load_model(self) -> None:
        started_at = time.perf_counter()
        weights = transformers_layout_weights(self._model_dir, self._repo_id)
        if weights is None:
            raise TranscriptionError(f"{self._repo_id} isn't downloaded, or isn't in transformers' layout")
        packed = packed_model_path(weights)
        if not packed.exists():
            _pack_snapshot(weights, packed)
        self._lib = _open_library(self._dll_path)
        error = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        handle = self._lib.npk_load(str(packed).encode(), error, _ERROR_BUFFER_BYTES)
        if not handle:
            raise TranscriptionError(f"Native Parakeet failed to load: {error.value.decode(errors='replace')}")
        self._handle = handle
        _logger.info(
            "[load_model native parakeet] complete in %.2fs | %s", time.perf_counter() - started_at, self._stats(),
        )

    def is_model_loaded(self) -> bool:
        return self._handle is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if self._handle is None or self._lib is None:
            raise TranscriptionError("Model not loaded")
        if audio.sample_rate != _SAMPLE_RATE:
            raise TranscriptionError(f"Native Parakeet needs {_SAMPLE_RATE} Hz audio, got {audio.sample_rate}")
        started_at = time.perf_counter()
        step = _CHUNK_SECONDS * _SAMPLE_RATE
        parts = [self._transcribe_piece(audio.data[start:start + step]) for start in range(0, audio.data.size, step)]
        return TranscriptionResult(
            text=" ".join(part for part in parts if part),
            language="en",
            language_probability=0.0,
            transcription_seconds=time.perf_counter() - started_at,
        )

    def dispose(self) -> None:
        if self._handle is None or self._lib is None:
            return
        self._lib.npk_free(self._handle)
        self._handle = None

    def _transcribe_piece(self, piece: np.ndarray) -> str:
        samples = np.ascontiguousarray(piece, dtype=np.float32)
        if samples.size < 2:
            return ""
        length = self._lib.npk_transcribe(
            self._handle, samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), samples.size,
            self._text, _TEXT_BUFFER_BYTES,
        )
        if length < 0:
            raise TranscriptionError(f"Native Parakeet failed: {self._text.value.decode(errors='replace')}")
        _logger.debug("native parakeet %s", self._stats())
        return self._text.raw[:length].decode("utf-8", errors="replace").strip()

    def _stats(self) -> dict:
        return json.loads(self._lib.npk_stats(self._handle).decode())


def _pack_snapshot(weights: Path, packed: Path) -> None:
    from skryb.transcription.parakeet_pack import pack

    _logger.info("Packing Parakeet for the native engine (once per download): %s", packed)
    started_at = time.perf_counter()
    pack(weights, packed)
    _logger.info("Packed in %.1fs", time.perf_counter() - started_at)


def _open_library(dll_path: Path) -> ctypes.CDLL:
    if not dll_path.exists():
        raise TranscriptionError(
            f"The native Parakeet engine isn't installed ({dll_path} is missing). "
            "Reinstall Skryb, or in a source checkout run native/parakeet/build.sh."
        )
    hip_bin = _rocm_runtime_dir()
    if hip_bin is not None:
        os.add_dll_directory(str(hip_bin))
    lib = ctypes.CDLL(str(dll_path))
    lib.npk_load.restype = ctypes.c_void_p
    lib.npk_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.npk_transcribe.restype = ctypes.c_int
    lib.npk_transcribe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.npk_stats.restype = ctypes.c_char_p
    lib.npk_stats.argtypes = [ctypes.c_void_p]
    lib.npk_free.restype = None
    lib.npk_free.argtypes = [ctypes.c_void_p]
    return lib


def _rocm_runtime_dir() -> Path | None:
    """The HIP runtime the DLL links against, from the ROCm SDK wheel, found without importing it."""
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))) / "bin"
