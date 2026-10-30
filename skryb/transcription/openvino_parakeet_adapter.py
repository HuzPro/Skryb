"""Parakeet TDT (Parakeet Ultra, or Phonon expanded to dense) with its encoder on an Intel NPU, through OpenVINO.

The encoder is nearly all of the work. It is exported once from the
transformers-layout weights to OpenVINO IR beside the download
(``snapshots/<sha>/.openvino``), rewritten with int8 weights
(``openvino_int8_weights``) and compiled for the NPU, or for the CPU when
there is no NPU or it refuses the graph. The mel features run in NumPy and
the greedy TDT decode on OpenVINO's CPU device (``openvino_tdt_decoder``):
they are small, and loading never imports torch once the export
(``openvino_parakeet_export``, the one step that needs it) is done.

The NPU compiles static shapes only, so features are padded up to one of
four fixed lengths and the padding is masked out the way NeMo masks a
padded batch: attention ignores padded keys, and the convolution module
sees zeros there. OpenVINO's ``CACHE_DIR`` keeps the compiled blobs, so
only the first load pays for compiling.
"""

import logging
import shutil
import stat
import sys
import time
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.backend_imports import make_engine_dependency_error
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.hf_model_manager import loadable_snapshot_dir
from skryb.transcription.openvino_int8_weights import compress_encoder
from skryb.transcription.openvino_tdt_decoder import OpenVinoTdtDecoder
from skryb.transcription.parakeet_loader import transformers_layout_weights
from skryb.transcription.parakeet_numpy import bundled_tokenizer, log_mel_features
from skryb.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_FRAMES_PER_SECOND = 100
_N_MELS = 128
# 68% of logged dictations fit in 10 s and 92% in 30 s. Longer audio is cut
# at a pause into pieces of at most the longest length. On a Core 5 320 NPU
# a pass takes 34 ms at 5 s, 62 at 10, 86 at 15 and 249 at 30 (3 s costs the
# same as 5), and each length compiled adds about 0.3 GB of memory and 0.4 GB
# of cached blob on disk.
BUCKET_SECONDS = (5, 10, 15, 30)
_PAUSE_SEARCH_SECONDS = 5
_PAUSE_WINDOW_SAMPLES = _SAMPLE_RATE // 10
_IR_DIRECTORY_NAME = ".openvino"
_EXPORTED_IR_FILENAME = "encoder.xml"
_IR_FILENAME = "encoder-int8.xml"
_DECODER_FILENAME = "decoder.npz"
# The STFT reflects 256 samples at each edge, so a piece needs more than that (16 ms).
_SHORTEST_SAMPLES = 256
_CACHE_DIRECTORY_NAME = "compiled-int8"
# Where 0.9.0 kept its fp16 blobs, gigabytes nothing reads any more.
_OLD_CACHE_DIRECTORY_NAME = "compiled"
_NPU = "NPU"
_CPU = "CPU"
# Without turbo the NPU clocks down between dictations, and a 10 s pass that
# takes 60 ms back to back takes 107-132 ms after a pause; with it, 62 ms
# either way. High priority keeps other NPU work, such as camera effects,
# from queueing ahead of dictation.
_NPU_CONFIG = {"NPU_TURBO": "YES", "MODEL_PRIORITY": "HIGH"}
_INSTALL_HINT = "The NPU engine needs OpenVINO. Install it with: pip install openvino"


class OpenVinoParakeetAdapter(TranscriptionEngine):
    """Parakeet TDT (parakeet-ultra) with the encoder on the NPU."""

    def __init__(self, model_dir: Path, repo_id: str) -> None:
        self._model_dir = model_dir
        self._repo_id = repo_id
        self._model: _NpuParakeet | None = None
        self._blas_threads_before: int | None = None

    def load_model(self) -> None:
        started_at = time.monotonic()
        openvino = import_openvino()
        export_dir = loadable_snapshot_dir(self._model_dir, self._repo_id) / _IR_DIRECTORY_NAME
        ir_path, decoder_path = export_dir / _IR_FILENAME, export_dir / _DECODER_FILENAME
        exported = export_dir / _EXPORTED_IR_FILENAME
        # The export writes only what is missing, so an existing int8 encoder stands in for the fp16 one.
        encoder_source = ir_path if ir_path.is_file() else exported
        if not (decoder_path.is_file() and encoder_source.is_file()):
            self._export(openvino, encoder_source, decoder_path)
        if not ir_path.is_file():
            _write_int8_encoder(openvino, exported, ir_path)
        encoder = compile_encoder(openvino, ir_path, _devices_to_try(openvino))
        with np.load(decoder_path) as tensors:
            decoder = OpenVinoTdtDecoder(openvino, dict(tensors))
        if self._blas_threads_before is None:
            self._blas_threads_before = set_numpy_blas_threads(1)
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
        if self._blas_threads_before is not None:
            set_numpy_blas_threads(self._blas_threads_before)
            self._blas_threads_before = None


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
    def __init__(self, encoder: CompiledEncoder, decoder: OpenVinoTdtDecoder, tokenizer: object) -> None:
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


def _write_int8_encoder(openvino: object, exported: Path, ir_path: Path) -> None:
    """The exported fp16 encoder rewritten with int8 weights; the fp16 copy and 0.9.0's blobs are then deleted."""
    started_at = time.monotonic()
    layers = compress_encoder(openvino, exported, ir_path)
    exported.unlink()
    exported.with_suffix(".bin").unlink(missing_ok=True)
    delete_cache_directory(ir_path.parent / _OLD_CACHE_DIRECTORY_NAME)
    _logger.info("Wrote the encoder with int8 weights on %d layers in %.1fs", layers, time.monotonic() - started_at)


def delete_cache_directory(directory: Path) -> None:
    """Delete *directory*, whose blobs OpenVINO marks read-only; a failure is logged, never raised."""
    try:
        _delete_read_only_tree(directory)
    except OSError:
        _logger.warning("Could not delete the old compiled encoder in %s", directory, exc_info=True)


def _delete_read_only_tree(directory: Path) -> None:
    if not directory.is_dir():
        return
    for path in directory.rglob("*"):
        path.chmod(stat.S_IWRITE | stat.S_IREAD)
    shutil.rmtree(directory)


def set_numpy_blas_threads(count: int) -> int | None:
    """Set NumPy's OpenBLAS thread count; the count it had, or None when it can't be reached.

    OpenBLAS keeps its threads spinning for a while after each product, and
    on a six-core laptop the six it starts for the mel features fight the
    decoder's two: decoding 6 s took 160-260 ms instead of 50-100. So the
    engine runs it on one thread while loaded.
    """
    try:
        return _swap_openblas_threads(count)
    except (OSError, AttributeError, IndexError):
        _logger.debug("NumPy's OpenBLAS thread count could not be set", exc_info=True)
        return None


def _swap_openblas_threads(count: int) -> int:
    import ctypes

    libs = Path(np.__file__).parent.with_name("numpy.libs")
    library = ctypes.CDLL(str(sorted(libs.glob("libscipy_openblas*"))[0]))
    library.scipy_openblas_get_num_threads64_.restype = ctypes.c_int
    before = library.scipy_openblas_get_num_threads64_()
    library.scipy_openblas_set_num_threads64_(ctypes.c_int(count))
    return before


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

    A warm-up on silence at every length must come back finite: a device
    that computes in fp16 and overflows is caught here and the next device
    is tried. It also takes each length's first-run cost off the first
    dictation.
    """
    core = openvino.Core()
    config = {
        "CACHE_DIR": str(ir_path.parent / _CACHE_DIRECTORY_NAME),
        "PERFORMANCE_HINT": "LATENCY",
        **(_NPU_CONFIG if device == _NPU else {}),
    }
    compiled_by_frames = {}
    for frames in ([bucket_frames(s) for s in BUCKET_SECONDS] if padded else [0]):
        model = core.read_model(str(ir_path))
        if frames:
            model.reshape({"features": [1, _N_MELS, frames], "length": [1]})
        compiled_by_frames[frames] = core.compile_model(model, device, config)
    encoder = CompiledEncoder(device, compiled_by_frames)
    for frames in compiled_by_frames:
        warm = encoder(np.zeros((1, _N_MELS, frames or bucket_frames(1)), np.float32))
        if not np.isfinite(warm).all():
            raise RuntimeError(f"the encoder gave non-finite values on {device}")
    return encoder
