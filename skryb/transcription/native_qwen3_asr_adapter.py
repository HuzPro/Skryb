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
from collections.abc import Callable
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.native_engine_dll import SHIPPED_DLL, native_engine_dll
from skryb.transcription.qwen3_asr_prompt import prompt_text
from skryb.transcription.result import TranscriptionResult
from skryb.transcription.roman_urdu import romanize


_logger = logging.getLogger(__name__)

_PACK_DIRECTORY_NAME = ".qasr"
_PACK_FILENAME = "qwen3-asr.bin"
_SAMPLE_RATE = 16000
_TEXT_BUFFER_BYTES = 1 << 20
_ERROR_BUFFER_BYTES = 1024


def packed_model_path(snapshot: Path) -> Path:
    """Where the packed engine file for *snapshot* lives."""
    return snapshot / _PACK_DIRECTORY_NAME / _PACK_FILENAME


class NativeQwen3AsrAdapter(TranscriptionEngine):
    """Batch Qwen3-ASR through ``qasr.dll``. One call at a time, like every adapter here."""

    def __init__(self, snapshot: Path, dll_path: Path | None = None) -> None:
        self._snapshot = snapshot
        self._dll_path = dll_path or native_engine_dll() or SHIPPED_DLL
        self._lib: ctypes.CDLL | None = None
        self._handle: int | None = None
        self._text = ctypes.create_string_buffer(_TEXT_BUFFER_BYTES)
        self._tokenizer: Callable[[str], list[int]] | None = None
        self._warned_unprompted = False

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
        lib, handle, samples = self._ready(audio)
        return self._decoded(lambda: lib.qasr_transcribe(
            handle, samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), samples.size,
            self._text, _TEXT_BUFFER_BYTES,
        ))

    def transcribe_prompted(
        self, audio: AudioSegment, context: str | None, language: str | None,
    ) -> TranscriptionResult:
        """Transcribe with names to spell and a forced language, or detected when *language* is None."""
        lib, handle, samples = self._ready(audio)
        if not hasattr(lib, "qasr_transcribe_prompted"):
            self._warn_unprompted_once()
            return self.transcribe(audio)
        prefix, suffix = (np.array(self._token_ids(text), np.int32) for text in prompt_text(context, language))
        ints = ctypes.POINTER(ctypes.c_int)
        return self._decoded(lambda: lib.qasr_transcribe_prompted(
            handle, samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), samples.size,
            prefix.ctypes.data_as(ints), prefix.size, suffix.ctypes.data_as(ints), suffix.size,
            self._text, _TEXT_BUFFER_BYTES,
        ))

    def _ready(self, audio: AudioSegment) -> tuple[ctypes.CDLL, int, np.ndarray]:
        if self._handle is None or self._lib is None:
            raise TranscriptionError("Model not loaded")
        if audio.sample_rate != _SAMPLE_RATE:
            raise TranscriptionError(f"Native Qwen3-ASR needs {_SAMPLE_RATE} Hz audio, got {audio.sample_rate}")
        return self._lib, self._handle, np.ascontiguousarray(audio.data, dtype=np.float32)

    def _decoded(self, run: Callable[[], int]) -> TranscriptionResult:
        started_at = time.perf_counter()
        length = run()
        if length < 0:
            raise TranscriptionError(f"Native Qwen3-ASR failed: {self._text.value.decode(errors='replace')}")
        _logger.debug("native qwen3-asr %s", self._stats())
        return TranscriptionResult(
            # Qwen3-ASR has no Urdu: it writes spoken Urdu in Hindi script.
            text=romanize(self._text.raw[:length].decode("utf-8", errors="replace")),
            language="en",
            language_probability=0.0,
            transcription_seconds=time.perf_counter() - started_at,
        )

    def _token_ids(self, text: str) -> list[int]:
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            tokenizer = Tokenizer.from_file(str(self._snapshot / "tokenizer.json"))
            self._tokenizer = lambda piece: tokenizer.encode(piece, add_special_tokens=False).ids
        return self._tokenizer(text)

    def _warn_unprompted_once(self) -> None:
        if self._warned_unprompted:
            return
        self._warned_unprompted = True
        _logger.warning(
            "This qasr.dll predates prompts: meeting names and the language setting are not passed on. "
            "Rebuild it with native/engine/build.sh.",
        )

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
    if hasattr(lib, "qasr_transcribe_prompted"):
        lib.qasr_transcribe_prompted.restype = ctypes.c_int
        ints = ctypes.POINTER(ctypes.c_int)
        lib.qasr_transcribe_prompted.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ints, ctypes.c_int, ints, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
    lib.qasr_stats.restype = ctypes.c_char_p
    lib.qasr_stats.argtypes = [ctypes.c_void_p]
    lib.qasr_free.restype = None
    lib.qasr_free.argtypes = [ctypes.c_void_p]
    return lib


def _rocm_runtime_dir() -> Path | None:
    """The HIP runtime the DLL links against, from the ROCm SDK wheel, found without importing it."""
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))) / "bin"
