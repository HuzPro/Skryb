"""Nemotron 3 Diarization on the CPU, through NVIDIA's NeMo-Speech.cpp (experimental).

For computers with neither a GPU nor an Intel NPU for the other diarizers. It streams like they do, one label per
10 ms, and NVIDIA's q8_0 GGUF of the same checkpoint keeps up with live audio on a desktop CPU (about 3x real time on
six cores). When it falls behind, the meeting's labels catch up after it ends. The library is the C ABI in
``nemo_speech/diar.h``, from NVIDIA's own Windows CPU release.
"""

import ctypes
import logging
import time
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.meetings.capture import RATE
from skryb.meetings.segmenter import SILENCE
from skryb.transcription.native_engine_dll import nemo_speech_library

_logger = logging.getLogger(__name__)

_SPEECH = 0.5
_CPU = -1
# Keeps the model's own geometry: the override fields are "<= 0 keeps the preset", but left context keeps it only
# below 0.
_KEEP = 0
_KEEP_LEFT_CONTEXT = -1


class _ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model_path", ctypes.c_char_p),
        ("gpu", ctypes.c_int32),
        ("preset", ctypes.c_char_p),
        ("chunk_frames", ctypes.c_int32),
        ("right_context_frames", ctypes.c_int32),
        ("left_context_frames", ctypes.c_int32),
        ("fifo_frames", ctypes.c_int32),
        ("spkcache_frames", ctypes.c_int32),
        ("update_period_frames", ctypes.c_int32),
    ]


class CpuDiarizer:
    """Who is talking, one label per frame (a speaker index or SILENCE), from NeMo-Speech.cpp on the CPU."""

    def __init__(self, model: Path, library: Path | None = None) -> None:
        self._lib = _open_library(library or nemo_speech_library())
        started_at = time.perf_counter()
        self._model = ctypes.c_void_p()
        config = _ModelConfig(ctypes.sizeof(_ModelConfig), _path_bytes(model), _CPU, None,
                              _KEEP, _KEEP, _KEEP_LEFT_CONTEXT, _KEEP, _KEEP, _KEEP)
        self._check(self._lib.nemo_speech_diar_create(ctypes.byref(config), ctypes.byref(self._model)), "load")
        self._stream = ctypes.c_void_p()
        self._check(self._lib.nemo_speech_diar_stream_open(self._model, ctypes.byref(self._stream)), "start")
        self._speakers = int(self._lib.nemo_speech_diar_num_speakers(self._model))
        self.frame_seconds = float(self._lib.nemo_speech_diar_seconds_per_frame(self._model))
        self._labelled = 0
        _logger.info("[load_model CPU diarizer] complete in %.2fs | NeMo-Speech.cpp, %d speakers, %.0f ms frames",
                     time.perf_counter() - started_at, self._speakers, self.frame_seconds * 1000)

    def push(self, audio: np.ndarray, last: bool = False) -> list[int]:
        samples = np.ascontiguousarray(audio, dtype=np.float32)
        if len(samples):
            self._check(self._lib.nemo_speech_diar_stream_push_f32(
                self._stream, samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(samples), RATE), "run")
        if last:
            self._check(self._lib.nemo_speech_diar_stream_finish(self._stream), "finish")
        return self._new_labels()

    def dispose(self) -> None:
        if self._stream:
            self._lib.nemo_speech_diar_stream_close(self._stream)
            self._stream = ctypes.c_void_p()
        if self._model:
            self._lib.nemo_speech_diar_destroy(self._model)
            self._model = ctypes.c_void_p()

    def _new_labels(self) -> list[int]:
        """Labels for the frames finished since the last call: the strongest speaker above 0.5, else SILENCE."""
        count = int(self._lib.nemo_speech_diar_frame_count(self._stream))
        if count <= self._labelled:
            return []
        start = int(self._lib.nemo_speech_diar_frame_probs_start(self._stream))
        # ponytail: copies every retained frame (up to ~20 min, ~4 MB) once per chunk; copy only the tail if it shows.
        probs = np.empty(((count - start), self._speakers), np.float32)
        self._check(self._lib.nemo_speech_diar_frame_probs(
            self._stream, probs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), probs.size), "read")
        # Frames the library already folded into segments were read on an earlier call; this keeps the timeline whole
        # if one ever wasn't.
        lost = max(0, start - self._labelled)
        fresh = probs[max(0, self._labelled - start):]
        self._labelled = count
        labels = np.where(fresh.max(axis=-1) > _SPEECH, fresh.argmax(axis=-1), SILENCE)
        return [SILENCE] * lost + labels.tolist()

    def _check(self, status: int, step: str) -> None:
        if status != 0:
            detail = self._lib.nemo_speech_asr_last_error() or b""
            raise TranscriptionError(
                f"The CPU diarizer failed to {step} (status {status}): {detail.decode(errors='replace')}")


def _open_library(path: Path | None) -> ctypes.CDLL:
    if path is None or not path.is_file():
        raise TranscriptionError(f"The CPU diarizer isn't installed ({path or 'nemo_speech_asr_c.dll'} is missing).")
    try:
        lib = _bind(ctypes.CDLL(str(path)))
    except (OSError, AttributeError) as error:
        raise TranscriptionError(f"The CPU diarizer didn't load from {path}: {error}") from error
    return lib


def _bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    handle, status, floats = ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
    for name, restype, argtypes in (
        ("nemo_speech_diar_create", status, [ctypes.POINTER(_ModelConfig), ctypes.POINTER(handle)]),
        ("nemo_speech_diar_destroy", None, [handle]),
        ("nemo_speech_diar_num_speakers", ctypes.c_int32, [handle]),
        ("nemo_speech_diar_seconds_per_frame", ctypes.c_double, [handle]),
        ("nemo_speech_diar_stream_open", status, [handle, ctypes.POINTER(handle)]),
        ("nemo_speech_diar_stream_push_f32", status, [handle, floats, ctypes.c_size_t, ctypes.c_int32]),
        ("nemo_speech_diar_stream_finish", status, [handle]),
        ("nemo_speech_diar_stream_close", None, [handle]),
        ("nemo_speech_diar_frame_count", ctypes.c_int64, [handle]),
        ("nemo_speech_diar_frame_probs_start", ctypes.c_int64, [handle]),
        ("nemo_speech_diar_frame_probs", status, [handle, floats, ctypes.c_size_t]),
        ("nemo_speech_asr_last_error", ctypes.c_char_p, []),
    ):
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, argtypes
    return lib


def _path_bytes(path: Path) -> bytes:
    return str(path).encode()
