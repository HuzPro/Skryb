"""CrisperWhisper 2.0 (Large, Turbo) on the native HIP engine (``native/whisper``), no torch or Transformers.

The engine is a DLL of hand-written kernels for the RX 9060 XT (gfx1200). It copies ``model.safetensors``
into VRAM as it is on disk, so nothing is converted or written next to the download. The prompt and the
transcript text go through the model's own ``tokenizer.json``, the same way the CTranslate2 adapter does.
"""

import ctypes
import json
import logging
import os
import struct
import time
from pathlib import Path

import numpy as np

from skryb.core.exceptions import GpuEngineError, TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.engine_types import CrisperWhisperMode
from skryb.transcription.native_engine_dll import native_whisper_dll, rocm_runtime_dir
from skryb.transcription.result import TranscriptionResult
from skryb.transcription.whisper_features import transcribe_in_windows
from skryb.transcription.whisper_tokens import WhisperTokens

_logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_WINDOW_SAMPLES = 30 * _SAMPLE_RATE
_DECODER_POSITIONS = 448
_ERROR_BUFFER_BYTES = 1024
_WEIGHTS = "model.safetensors"
# The kernels are written for this shape: Whisper large-v2/v3 and large-v3-turbo.
_SHAPE = {
    "d_model": 1280, "encoder_attention_heads": 20, "decoder_attention_heads": 20,
    "encoder_ffn_dim": 5120, "decoder_ffn_dim": 5120,
    "max_source_positions": 1500, "max_target_positions": _DECODER_POSITIONS,
}


class _CwConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int) for name in ("n_mels", "encoder_layers", "decoder_layers", "vocab", "eos")]


class NativeCrisperWhisperAdapter(TranscriptionEngine):
    """Greedy CrisperWhisper through ``cw.dll``, with Nyra's intended/verbatim prompt. One call at a time."""

    def __init__(
        self,
        model_dir: Path,
        language: str,
        mode: CrisperWhisperMode = "intended",
        dll_path: Path | None = None,
        cuda: bool = False,
    ) -> None:
        self._model_dir = model_dir
        self._language = language
        self._mode = mode
        self._dll_path = dll_path or native_whisper_dll(cuda=cuda) or Path("cw_cuda.dll" if cuda else "cw.dll")
        self._cuda = cuda
        self._lib: ctypes.CDLL | None = None
        self._handle: int | None = None
        self._tokens: WhisperTokens | None = None
        self._ids = (ctypes.c_int * _DECODER_POSITIONS)()
        self._error = ctypes.create_string_buffer(_ERROR_BUFFER_BYTES)

    def load_model(self) -> None:
        started_at = time.perf_counter()
        self.dispose()
        tokens = WhisperTokens(self._model_dir)
        weights = self._model_dir / _WEIGHTS
        names, offsets, vocab = _tensor_offsets(weights)
        config = _engine_config(self._model_dir, vocab)
        suppress = _suppressed_tokens(self._model_dir)
        if self._cuda:
            _logger.warning("Running the experimental NVIDIA build of the native engine: %s", self._dll_path.name)
        self._lib = _open_library(self._dll_path)
        encoded = [name.encode() for name in names]
        handle = self._lib.cw_load(
            str(weights).encode(),
            (ctypes.c_char_p * len(encoded))(*encoded),
            (ctypes.c_uint64 * len(offsets))(*offsets),
            len(names),
            ctypes.byref(config),
            (ctypes.c_int * len(suppress))(*suppress),
            len(suppress),
            self._error,
            _ERROR_BUFFER_BYTES,
        )
        if not handle:
            raise GpuEngineError(f"Native CrisperWhisper failed to load: {self._error_text()}")
        self._handle = handle
        self._tokens = tokens
        _logger.info(
            "[load_model native crisperwhisper %s] complete in %.2fs | %s",
            self._model_dir.parent.parent.name, time.perf_counter() - started_at, self._stats(),
        )

    def is_model_loaded(self) -> bool:
        return self._handle is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if self._handle is None:
            raise TranscriptionError("Model not loaded")
        if audio.sample_rate != _SAMPLE_RATE:
            raise TranscriptionError(f"Native CrisperWhisper needs {_SAMPLE_RATE} Hz audio, got {audio.sample_rate}")
        started_at = time.perf_counter()
        text = transcribe_in_windows(audio, self._generate_text)
        return TranscriptionResult(
            text=text,
            language=self._language,
            language_probability=0.0,
            transcription_seconds=time.perf_counter() - started_at,
        )

    def dispose(self) -> None:
        if self._handle is None or self._lib is None:
            return
        self._lib.cw_free(self._handle)
        self._handle = None
        self._tokens = None

    def _generate_text(self, window: AudioSegment, context: str) -> str:
        if self._handle is None or self._lib is None or self._tokens is None:
            raise TranscriptionError("Model not loaded")
        prompt = self._tokens.decoder_prompt(self._language, self._mode, context)
        samples = np.ascontiguousarray(window.data[:_WINDOW_SAMPLES], dtype=np.float32)
        count = self._lib.cw_transcribe(
            self._handle,
            samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            samples.size,
            (ctypes.c_int * len(prompt))(*prompt),
            len(prompt),
            self._ids,
            _DECODER_POSITIONS,
            self._error,
            _ERROR_BUFFER_BYTES,
        )
        if count < 0:
            raise TranscriptionError(f"Native CrisperWhisper failed: {self._error_text()}")
        _logger.debug("native crisperwhisper %s", self._stats())
        return self._tokens.decode(self._ids[:count])

    def _stats(self) -> dict:
        return json.loads(self._lib.cw_stats(self._handle).decode())

    def _error_text(self) -> str:
        return self._error.value.decode(errors="replace")


def _tensor_offsets(weights: Path) -> tuple[list[str], list[int], int]:
    """Each model tensor's file offset, and the vocabulary size from the token embedding's rows."""
    if not weights.is_file():
        raise TranscriptionError(f"{weights} is missing; download the model again")
    with weights.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    names, offsets = [], []
    for name, info in header.items():
        if not name.startswith("model."):
            continue
        if info["dtype"] != "BF16":
            raise TranscriptionError(f"The native engine needs bf16 weights; {name} is {info['dtype']}")
        names.append(name)
        offsets.append(8 + length + info["data_offsets"][0])
    if len({offset % 16 for offset in offsets}) > 1:
        raise TranscriptionError("The weights in model.safetensors aren't laid out the same way")
    embed = header.get("model.decoder.embed_tokens.weight")
    if embed is None:
        raise TranscriptionError("model.safetensors has no token embedding")
    return names, offsets, int(embed["shape"][0])


def _engine_config(model_dir: Path, vocab: int) -> _CwConfig:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    for key, expected in _SHAPE.items():
        if config.get(key) != expected:
            raise TranscriptionError(f"The native engine can't run this Whisper: {key} is {config.get(key)}")
    return _CwConfig(
        n_mels=int(config["num_mel_bins"]),
        encoder_layers=int(config["encoder_layers"]),
        decoder_layers=int(config["decoder_layers"]),
        vocab=vocab,
        eos=int(config["eos_token_id"]),
    )


def _suppressed_tokens(model_dir: Path) -> list[int]:
    """The generation config's suppress_tokens, which the Transformers adapter passes to generate()."""
    generation = json.loads((model_dir / "generation_config.json").read_text(encoding="utf-8"))
    return [int(token) for token in generation.get("suppress_tokens") or [] if int(token) >= 0]


def _open_library(dll_path: Path) -> ctypes.CDLL:
    if not dll_path.exists():
        raise GpuEngineError(
            f"The native CrisperWhisper engine isn't installed ({dll_path} is missing). "
            "Reinstall Skryb, or in a source checkout run native/whisper/build.sh (native/cuda/build.py for NVIDIA)."
        )
    hip_bin = rocm_runtime_dir()
    if hip_bin is not None:
        os.add_dll_directory(str(hip_bin))
    lib = ctypes.CDLL(str(dll_path))
    lib.cw_load.restype = ctypes.c_void_p
    lib.cw_load.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_uint64), ctypes.c_int,
        ctypes.POINTER(_CwConfig), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    lib.cw_transcribe.restype = ctypes.c_int
    lib.cw_transcribe.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
    ]
    lib.cw_stats.restype = ctypes.c_char_p
    lib.cw_stats.argtypes = [ctypes.c_void_p]
    lib.cw_free.restype = None
    lib.cw_free.argtypes = [ctypes.c_void_p]
    return lib
