"""Qwen3-ASR-1.7B on the native HIP engine (``native/engine``), with no torch in the process.

The engine is a DLL of hand-written kernels for the RX 9060 XT (gfx1200).
It reads one packed file laid out exactly as it sits in VRAM. That file is
built once from the downloaded snapshot, inside the snapshot it came from
(``snapshots/<sha>/.qasr``), the way Phonon-2 unpacks next to its archive.
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
from skryb.transcription.native_engine_dll import SHIPPED_DLL, native_engine_dll
from skryb.transcription.result import TranscriptionResult
from skryb.transcription.roman_urdu import romanize


_logger = logging.getLogger(__name__)

_PACK_DIRECTORY_NAME = ".qasr"
_PACK_FILENAME = "qwen3-asr.bin"
_SAMPLE_RATE = 16000
_TEXT_BUFFER_BYTES = 1 << 20
_ERROR_BUFFER_BYTES = 1024
# The engine's own odds of "?" over "." where it ended the transcript with a full stop. On 1,108 of
# Huz's dictations, 0.3 turned 6 full stops into question marks: 2 clearly right, none clearly
# wrong, 4 that read fine either way. Lower bars caught more but got about 3 in 10 wrong.
QUESTION_THRESHOLD = 0.3


def packed_model_path(snapshot: Path) -> Path:
    """Where the packed engine file for *snapshot* lives."""
    return snapshot / _PACK_DIRECTORY_NAME / _PACK_FILENAME


class NativeQwen3AsrAdapter(TranscriptionEngine):
    """Batch Qwen3-ASR through ``qasr.dll``. One call at a time, like every adapter here."""

    def __init__(
        self, snapshot: Path, dll_path: Path | None = None, question_threshold: float | None = QUESTION_THRESHOLD,
    ) -> None:
        self._snapshot = snapshot
        self._dll_path = dll_path or native_engine_dll() or SHIPPED_DLL
        self._question_threshold = question_threshold
        self._lib: ctypes.CDLL | None = None
        self._handle: int | None = None
        self._text = ctypes.create_string_buffer(_TEXT_BUFFER_BYTES)

    def load_model(self) -> None:
        started_at = time.perf_counter()
        packed = packed_model_path(self._snapshot)
        if not packed.exists():
            self._pack_snapshot(packed)
        self._lib = _open_library(self._dll_path)
        error = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)
        handle = self._lib.qasr_load(str(packed).encode(), error, _ERROR_BUFFER_BYTES)
        if not handle:
            raise TranscriptionError(f"Native Qwen3-ASR failed to load: {error.value.decode(errors='replace')}")
        self._handle = handle
        _logger.info(
            "[load_model native qwen3-asr] complete in %.2fs | %s",
            time.perf_counter() - started_at, self._stats(),
        )

    def is_model_loaded(self) -> bool:
        return self._handle is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if self._handle is None or self._lib is None:
            raise TranscriptionError("Model not loaded")
        if audio.sample_rate != _SAMPLE_RATE:
            raise TranscriptionError(f"Native Qwen3-ASR needs {_SAMPLE_RATE} Hz audio, got {audio.sample_rate}")
        samples = np.ascontiguousarray(audio.data, dtype=np.float32)
        started_at = time.perf_counter()
        length = self._lib.qasr_transcribe(
            self._handle, samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), samples.size,
            self._text, _TEXT_BUFFER_BYTES,
        )
        if length < 0:
            raise TranscriptionError(f"Native Qwen3-ASR failed: {self._text.value.decode(errors='replace')}")
        _logger.debug("native qwen3-asr %s", self._stats())
        text = self._text.raw[:length].decode("utf-8", errors="replace")
        return TranscriptionResult(
            # Qwen3-ASR has no Urdu: it writes spoken Urdu in Hindi script.
            text=romanize(self._with_heard_question(text)),
            language="en",
            language_probability=0.0,
            transcription_seconds=time.perf_counter() - started_at,
        )

    def _with_heard_question(self, text: str) -> str:
        """The final full stop as a question mark when the engine nearly chose one."""
        if self._question_threshold is None:
            return text
        core = text.rstrip()
        if not core.endswith(".") or core.endswith(".."):
            return text
        probability = self._lib.qasr_question_probability(self._handle)
        if probability < self._question_threshold:
            return text
        _logger.info("Final full stop heard as a question (p=%.2f).", probability)
        return core[:-1] + "?" + text[len(core):]

    def dispose(self) -> None:
        if self._handle is None or self._lib is None:
            return
        self._lib.qasr_free(self._handle)
        self._handle = None

    def _pack_snapshot(self, packed: Path) -> None:
        from skryb.transcription.qwen3_asr_pack import pack

        _logger.info("Packing Qwen3-ASR for the native engine (once per download): %s", packed)
        started_at = time.perf_counter()
        pack(self._snapshot, packed)
        _logger.info("Packed in %.1fs", time.perf_counter() - started_at)

    def _stats(self) -> dict:
        return json.loads(self._lib.qasr_stats(self._handle).decode())


def _open_library(dll_path: Path) -> ctypes.CDLL:
    if not dll_path.exists():
        raise TranscriptionError(
            f"The native Qwen3-ASR engine isn't installed ({dll_path} is missing). "
            "Reinstall Skryb, or in a source checkout run native/engine/build.sh."
        )
    hip_bin = _rocm_runtime_dir()
    if hip_bin is not None:
        os.add_dll_directory(str(hip_bin))
    lib = ctypes.CDLL(str(dll_path))
    lib.qasr_load.restype = ctypes.c_void_p
    lib.qasr_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.qasr_transcribe.restype = ctypes.c_int
    lib.qasr_transcribe.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.qasr_stats.restype = ctypes.c_char_p
    lib.qasr_stats.argtypes = [ctypes.c_void_p]
    lib.qasr_question_probability.restype = ctypes.c_float
    lib.qasr_question_probability.argtypes = [ctypes.c_void_p]
    lib.qasr_free.restype = None
    lib.qasr_free.argtypes = [ctypes.c_void_p]
    return lib


def _rocm_runtime_dir() -> Path | None:
    """The HIP runtime the DLL links against, from the ROCm SDK wheel, found without importing it."""
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))) / "bin"
