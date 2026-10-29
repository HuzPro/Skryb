"""Parakeet TDT (Parakeet Ultra, or Phonon expanded to dense) with its encoder on an Intel NPU, through OpenVINO.

The encoder is nearly all of the work. It is exported once from the
transformers-layout weights to OpenVINO IR beside the download
(``snapshots/<sha>/.openvino``) and compiled for the NPU, or for the CPU
when there is no NPU or it refuses the graph. The mel features, the LSTM
prediction network and the greedy TDT decode run in NumPy on the CPU
(``parakeet_numpy``): they are small, and loading never imports torch once
the export (``openvino_parakeet_export``, the one step that needs it) is done.

The NPU compiles static shapes only, so features are padded up to one of
two fixed lengths and the padding is masked out the way NeMo masks a
padded batch: attention ignores padded keys, and the convolution module
sees zeros there. OpenVINO's ``CACHE_DIR`` keeps the compiled blobs, so
only the first load pays for compiling.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.backend_imports import make_engine_dependency_error
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.hf_model_manager import loadable_snapshot_dir
from skryb.transcription.parakeet_loader import transformers_layout_weights
from skryb.transcription.parakeet_numpy import TdtDecoder, bundled_tokenizer, log_mel_features
from skryb.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_FRAMES_PER_SECOND = 100
_N_MELS = 128
# 68% of logged dictations fit in 10 s and 92% in 30 s. Longer audio is cut
# at a pause into pieces of at most the longest length.
BUCKET_SECONDS = (10, 30)
_PAUSE_SEARCH_SECONDS = 5
_PAUSE_WINDOW_SAMPLES = _SAMPLE_RATE // 10
_IR_DIRECTORY_NAME = ".openvino"
_IR_FILENAME = "encoder.xml"
_DECODER_FILENAME = "decoder.npz"
# The STFT reflects 256 samples at each edge, so a piece needs more than that (16 ms).
_SHORTEST_SAMPLES = 256
_CACHE_DIRECTORY_NAME = "compiled"
_NPU = "NPU"
_CPU = "CPU"
_INSTALL_HINT = "The NPU engine needs OpenVINO. Install it with: pip install openvino"


class OpenVinoParakeetAdapter(TranscriptionEngine):
    """Parakeet TDT (Bark, parakeet-ultra) with the encoder on the NPU."""

    def __init__(self, model_dir: Path, repo_id: str) -> None:
        self._model_dir = model_dir
        self._repo_id = repo_id
        self._model: _NpuParakeet | None = None

    def load_model(self) -> None:
        started_at = time.monotonic()
        openvino = import_openvino()
        export_dir = loadable_snapshot_dir(self._model_dir, self._repo_id) / _IR_DIRECTORY_NAME
        ir_path, decoder_path = export_dir / _IR_FILENAME, export_dir / _DECODER_FILENAME
        if not (ir_path.is_file() and decoder_path.is_file()):
            self._export(openvino, ir_path, decoder_path)
        encoder = compile_encoder(openvino, ir_path, _devices_to_try(openvino))
        with np.load(decoder_path) as tensors:
            decoder = TdtDecoder(dict(tensors))
        self._model = _NpuParakeet(encoder, decoder, bundled_tokenizer())
        _logger.info(
            "[load_model openvino] encoder on %s, ready in %.2fs",
            encoder.device, time.monotonic() - started_at,
        )

    def _export(self, openvino: object, ir_path: Path, decoder_path: Path) -> None:
        from functools import partial

        from skryb.transcription import openvino_parakeet_export as exporter

        weights = transformers_layout_weights(self._model_dir, self._repo_id)
        if weights is not None:
            load_state = partial(exporter.transformers_layout_state, weights)
        else:
            load_state = partial(exporter.phonon_state, self._phonon_container())
        exporter.export(openvino, load_state, ir_path, decoder_path)

    def _phonon_container(self) -> Path:
        """A Phonon snapshot's ``model.fermion``, unpacked from its archive by Fermion if need be."""
        from skryb.transcription.phonon2_cpu_adapter import phonon_container

        return phonon_container(loadable_snapshot_dir(self._model_dir, self._repo_id))

    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def encoder_device(self) -> str | None:
        """The OpenVINO device the encoder runs on, once loaded: "NPU" or "CPU"."""
        return None if self._model is None else self._model.encoder_device

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if self._model is None:
            raise TranscriptionError("Model not loaded")
        start = time.monotonic()
        pieces = split_at_pauses(audio.data, int(max(BUCKET_SECONDS) * _SAMPLE_RATE))
        text = " ".join(part for part in map(self._model.transcribe, pieces) if part)
        return TranscriptionResult(
            text=text,
            language="en",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - start,
        )

    def dispose(self) -> None:
        self._model = None


class CompiledEncoder:
    """The encoder compiled for one device: per padded length, or for any length."""

    def __init__(self, device: str, compiled_by_frames: dict[int, object]) -> None:
        self.device = device
        self._compiled_by_frames = compiled_by_frames

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Encoder frames for ``[1, 128, T]`` features, padding dropped."""
        frames = features.shape[-1]
        size, compiled = self._compiled_for(frames)
        padded = np.zeros((1, _N_MELS, size), np.float32)
        padded[..., :frames] = features
        result = compiled([padded, np.array([frames], np.int64)])
        encoded, length = result[0], int(result[1][0])
        return encoded[0, :length]

    def _compiled_for(self, frames: int) -> tuple[int, object]:
        if 0 in self._compiled_by_frames:
            return frames, self._compiled_by_frames[0]
        fitting = [size for size in self._compiled_by_frames if size >= frames]
        if not fitting:
            raise TranscriptionError(f"{frames} feature frames exceed the longest compiled length")
        return min(fitting), self._compiled_by_frames[min(fitting)]


class _NpuParakeet:
    def __init__(self, encoder: CompiledEncoder, decoder: TdtDecoder, tokenizer: object) -> None:
        self._encoder = encoder
        self._decoder = decoder
        self._tokenizer = tokenizer

    @property
    def encoder_device(self) -> str:
        return self._encoder.device

    def transcribe(self, samples: np.ndarray) -> str:
        if samples.size <= _SHORTEST_SAMPLES:
            return ""
        encoded = self._encoder(log_mel_features(samples))
        return str(self._tokenizer.DecodeIds(self._decoder.decode(encoded))).strip()


def split_at_pauses(samples: np.ndarray, longest: int) -> list[np.ndarray]:
    """Cut *samples* into pieces of at most *longest*, each at its quietest 100 ms.

    The cut is looked for in the last five seconds before the limit, so a
    word is cut through only when the speaker never paused there.
    """
    pieces = []
    search = _PAUSE_SEARCH_SECONDS * _SAMPLE_RATE
    while len(samples) > longest:
        tail = samples[longest - search:longest]
        windows = len(tail) // _PAUSE_WINDOW_SAMPLES
        energy = np.square(tail[:windows * _PAUSE_WINDOW_SAMPLES]).reshape(windows, -1).sum(axis=1)
        cut = longest - search + (int(np.argmin(energy)) + 1) * _PAUSE_WINDOW_SAMPLES
        pieces.append(samples[:cut])
        samples = samples[cut:]
    if len(samples):
        pieces.append(samples)
    return pieces


def import_openvino() -> object:
    """OpenVINO, with its converter's usage reporting kept out.

    OpenVINO's model converter sends usage events to Google Analytics
    unless the user has opted out. With its telemetry package unimportable
    it falls back to its own no-op stub, so Skryb hides the package before
    OpenVINO first loads.
    """
    sys.modules.setdefault("openvino_telemetry", None)
    try:
        import openvino
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="openvino", install_hint=_INSTALL_HINT, underlying=exc,
        ) from exc
    return openvino


def bucket_frames(seconds: int) -> int:
    """Feature frames for *seconds* of audio: the STFT is centred, so one extra."""
    return seconds * _FRAMES_PER_SECOND + 1


def _devices_to_try(openvino: object) -> list[str]:
    return ([_NPU] if _NPU in openvino.Core().available_devices else []) + [_CPU]


def compile_encoder(openvino: object, ir_path: Path, devices: list[str]) -> CompiledEncoder:
    """The encoder compiled for the first device that runs it cleanly."""
    for device in devices:
        try:
            return compile_for(openvino, ir_path, device, padded=device == _NPU)
        except RuntimeError as refusal:
            _logger.warning("OpenVINO could not run the encoder on %s: %s", device, refusal)
    raise TranscriptionError(f"OpenVINO could not run the encoder on {', '.join(devices)}")


def compile_for(openvino: object, ir_path: Path, device: str, padded: bool) -> CompiledEncoder:
    """Compile for *device*, at each fixed length when *padded*, and check the output.

    A warm-up on silence must come back finite: a device that computes in
    fp16 and overflows is caught here and the next device is tried.
    """
    core = openvino.Core()
    config = {
        "CACHE_DIR": str(ir_path.parent / _CACHE_DIRECTORY_NAME),
        "PERFORMANCE_HINT": "LATENCY",
    }
    compiled_by_frames = {}
    for frames in ([bucket_frames(s) for s in BUCKET_SECONDS] if padded else [0]):
        model = core.read_model(str(ir_path))
        if frames:
            model.reshape({"features": [1, _N_MELS, frames], "length": [1]})
        compiled_by_frames[frames] = core.compile_model(model, device, config)
    encoder = CompiledEncoder(device, compiled_by_frames)
    warm = encoder(np.zeros((1, _N_MELS, bucket_frames(1)), np.float32))
    if not np.isfinite(warm).all():
        raise RuntimeError(f"the encoder gave non-finite values on {device}")
    return encoder
