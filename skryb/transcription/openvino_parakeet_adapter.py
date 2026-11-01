"""Parakeet TDT (Parakeet Ultra, or Phonon expanded to dense) with its encoder on an Intel NPU, through OpenVINO.

The encoder is nearly all of the work. It is exported once from the
transformers-layout weights to OpenVINO IR beside the download
(``snapshots/<sha>/.openvino``), rewritten with int8 weights
(``openvino_int8_weights``) and compiled for the NPU, or for the CPU when
there is no NPU or it refuses the graph. The mel features run in NumPy and
the greedy TDT decode on OpenVINO's CPU device (``openvino_tdt_decoder``):
they are small, and a decode is a few hundred tiny steps that would each
pay the NPU's dispatch cost. Loading never imports torch once the export
(``openvino_parakeet_export``, the one step that needs it) is done.

The NPU compiles static shapes only, so features are padded up to one of
four fixed lengths and the padding is masked out the way NeMo masks a
padded batch: attention ignores padded keys, and the convolution module
sees zeros there. OpenVINO's ``CACHE_DIR`` keeps the compiled blobs.

Loading is staged so a prepared model takes dictation in under a second:
the IR is read once, the 10 s length comes up first beside the decoder,
and the other lengths load in parallel behind it. Compiling from scratch
takes minutes (the first start, or after an OpenVINO or driver update), so
for Phonon its CPU engine takes dictation meanwhile and the NPU takes over
once every length is up.
"""

import logging
import shutil
import stat
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.backend_imports import make_engine_dependency_error
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.hf_model_manager import loadable_snapshot_dir
from skryb.transcription.native_engine_dll import native_tdt_dll
from skryb.transcription.openvino_int8_weights import compress_encoder, opset
from skryb.transcription.openvino_tdt_decoder import OpenVinoTdtDecoder
from skryb.transcription.parakeet_loader import transformers_layout_weights
from skryb.transcription.parakeet_numpy import bundled_tokenizer, log_mel_features
from skryb.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_FRAMES_PER_SECOND = 100
_HOP_SAMPLES = _SAMPLE_RATE // _FRAMES_PER_SECOND
# A second of features: enough to check a length that takes any.
_PROBE_FRAMES = 101
_N_MELS = 128
# The fixed lengths, in feature frames (10 ms each). The encoder subsamples 8x, and a length that comes
# out at a multiple of 64 encoder frames fills the NPU's matrix units: on a Core 5 320 a pass takes 24, 42,
# 66, 101 and 165 ms for these 5.1, 10.2, 15.4, 20.5 and 30.7 s, 17-26% less than at round seconds (63, 126,
# 188 and 376 frames). Each length takes about 0.45 GB of NPU memory. 68% of logged dictations fit in 10 s
# and 92% in 30 s; longer audio is cut at a pause into pieces of at most the longest length.
BUCKET_FRAMES = (512, 1024, 1536, 2048, 3072)
# Up first, since most dictations fit it.
FIRST_BUCKET_FRAMES = 1024
# On a Core 5 320 a prepared model's first length is up in 0.3-0.5 s and compiling one from scratch takes
# over a minute, so a first length not up by now is compiling.
NPU_STARTUP_SECONDS = 1.0
_PAUSE_SEARCH_SECONDS = 5
_PAUSE_WINDOW_SAMPLES = _SAMPLE_RATE // 10
_IR_DIRECTORY_NAME = ".openvino"
_EXPORTED_IR_FILENAME = "encoder.xml"
_IR_FILENAME = "encoder-int8.xml"
_DECODER_FILENAME = "decoder.npz"
# The STFT reflects 256 samples at each edge, so a piece needs more than that (16 ms).
_SHORTEST_SAMPLES = 256
CACHE_DIRECTORY_NAME = "compiled-npu"
# Where 0.9.0 kept its fp16 blobs, gigabytes nothing reads any more.
_OLD_CACHE_DIRECTORY_NAME = "compiled"
# 0.10's round-second lengths, likewise left behind.
_STALE_CACHE_DIRECTORY_NAMES = (_OLD_CACHE_DIRECTORY_NAME, "compiled-int8")
_NPU = "NPU"
_CPU = "CPU"
# Without turbo the NPU clocks down between dictations, and a 10 s pass that
# takes 60 ms back to back takes 107-132 ms after a pause; with it, 62 ms
# either way. High priority keeps other NPU work, such as camera effects,
# from queueing ahead of dictation.
_NPU_CONFIG = {"NPU_TURBO": "YES", "MODEL_PRIORITY": "HIGH"}
_INSTALL_HINT = "The NPU engine needs OpenVINO. Install it with: pip install openvino"
# A settings change during the startup load starts a second load. Converting the same files from both at
# once failed with "Access is denied": Windows can't replace a file OpenVINO already has open.
_ONE_LOAD_AT_A_TIME = threading.Lock()


class OpenVinoParakeetAdapter(TranscriptionEngine):
    """Parakeet TDT with the encoder on the NPU; for Phonon, its CPU engine stands in while the NPU compiles."""

    def __init__(self, model_dir: Path, repo_id: str) -> None:
        self._model_dir = model_dir
        self._repo_id = repo_id
        self._model: _NpuParakeet | None = None
        self._stand_in: TranscriptionEngine | None = None
        # Why dictation is not on the NPU, when it isn't: shown in Skryb rather than left to the log.
        self._note: str | None = None
        self._failure: Exception | None = None
        # Notified when an engine goes live or a bring-up gives up, so load_model returns on the first one up.
        self._changed = threading.Condition()
        self._npu_settled = False
        self._stand_in_settled = True
        # Held for a whole transcription, so the NPU never takes over, or is disposed, halfway through one.
        self._turn = threading.RLock()
        self._disposed = False
        self._cpu_parts: tuple[object, object, Path] | None = None
        self._blas_threads_before: int | None = None

    def load_model(self) -> None:
        started_at = time.monotonic()
        if not (self._export_dir() / _IR_FILENAME).is_file():
            # The conversion takes minutes and unpacks the archive the stand-in reads too, so the stand-in goes first.
            self._start_stand_in(wait=True)
        threading.Thread(target=self._bring_up_npu, daemon=True, name="npu-bring-up").start()
        self._wait_until(lambda: self.is_model_loaded() or self._npu_settled, NPU_STARTUP_SECONDS)
        if not self.is_model_loaded():
            self._start_stand_in(wait=False)
        self._wait_until(lambda: self.is_model_loaded() or (self._npu_settled and self._stand_in_settled))
        if not self.is_model_loaded():
            raise self._failure or TranscriptionError("The NPU engine did not load")
        _logger.info(
            "[load_model openvino] taking dictation on %s after %.2fs",
            self.encoder_device, time.monotonic() - started_at,
        )

    def _export_dir(self) -> Path:
        return loadable_snapshot_dir(self._model_dir, self._repo_id) / _IR_DIRECTORY_NAME

    def _wait_until(self, ready: Callable[[], bool], timeout: float | None = None) -> None:
        with self._changed:
            self._changed.wait_for(ready, timeout)

    def _announce(self) -> None:
        with self._changed:
            self._changed.notify_all()

    def _start_stand_in(self, *, wait: bool) -> None:
        """Phonon's CPU engine, loading beside the NPU: whichever is up first takes dictation."""
        self._stand_in_settled = False
        if wait:
            self._try_stand_in()
        else:
            threading.Thread(target=self._try_stand_in, daemon=True, name="cpu-stand-in").start()

    def _try_stand_in(self) -> None:
        try:
            self._load_stand_in()
        except Exception:  # noqa: BLE001 - the NPU may still come up
            _logger.warning("The CPU stand-in could not load", exc_info=True)
        finally:
            self._stand_in_settled = True
            self._announce()

    def _load_stand_in(self) -> None:
        stand_in = self._cpu_engine()
        if stand_in is None:
            return
        stand_in.load_model()
        with self._turn:
            if self._model is None and self._stand_in is None and not self._disposed:
                self._stand_in = stand_in
                return
        stand_in.dispose()

    def _cpu_engine(self) -> TranscriptionEngine | None:
        """Phonon's own CPU engine; Parakeet Ultra has none here, so it waits for the NPU."""
        if transformers_layout_weights(self._model_dir, self._repo_id) is not None:
            return None
        from skryb.transcription.phonon2_cpu_adapter import Phonon2CpuAdapter

        return Phonon2CpuAdapter(loadable_snapshot_dir(self._model_dir, self._repo_id))

    def _bring_up_npu(self) -> None:
        try:
            with _ONE_LOAD_AT_A_TIME:
                self._bring_up_in_stages()
        except Exception as failure:  # noqa: BLE001 - surfaced by load_model, or noted beside the stand-in
            self._record_failure(failure)
        finally:
            self._npu_settled = True
            self._announce()

    def _bring_up_in_stages(self) -> None:
        started_at = time.monotonic()
        openvino = import_openvino()
        ir_path, decoder_path = self._converted_once(openvino)
        core = openvino.Core()
        cache_dir = ir_path.parent / CACHE_DIRECTORY_NAME
        with ThreadPoolExecutor(2) as side:
            # The NPU driver's start-up, about 0.2 s, happens beside reading the encoder instead of after it.
            woken = side.submit(_wake_npu, openvino, core, cache_dir)
            decoder = side.submit(_load_decoder, openvino, decoder_path)
            model = core.read_model(str(ir_path))
            self._cpu_parts = (core, model, cache_dir)
            npu_problem = woken.result()
            compiling_from = time.monotonic()
            encoder = self._first_encoder(core, model, cache_dir, npu_problem)
            first_seconds = time.monotonic() - compiling_from
            npu = _NpuParakeet(encoder, decoder.result(), bundled_tokenizer())
        on_npu = encoder.device == _NPU
        self._go_live(npu, every_length=not on_npu)
        if not on_npu or not self._passes_check_on_silence(npu, encoder):
            return
        # A first length that compiled rather than loaded means the rest compile too: one at a time leaves the
        # cores and memory to the stand-in taking dictation meanwhile.
        self._add_other_lengths(encoder, core, model, cache_dir, cold=first_seconds > NPU_STARTUP_SECONDS)
        self._go_live(npu, every_length=True)
        _logger.info(
            "[load_model openvino] %d lengths up on %s after %.2fs",
            encoder.length_count, encoder.device, time.monotonic() - started_at,
        )
        for stale in _STALE_CACHE_DIRECTORY_NAMES:
            delete_cache_directory(cache_dir.parent / stale)
        _delete_staged_leftovers(cache_dir.parent)

    def _converted_once(self, openvino: object) -> tuple[Path, Path]:
        """The int8 encoder IR and the decoder tensors, exported the first time only."""
        export_dir = self._export_dir()
        ir_path, decoder_path = export_dir / _IR_FILENAME, export_dir / _DECODER_FILENAME
        exported = export_dir / _EXPORTED_IR_FILENAME
        # The export writes only what is missing, so an existing int8 encoder stands in for the fp16 one.
        encoder_source = ir_path if ir_path.is_file() else exported
        if not (decoder_path.is_file() and encoder_source.is_file()):
            self._export(openvino, encoder_source, decoder_path)
        if not ir_path.is_file():
            _write_int8_encoder(openvino, exported, ir_path)
        return ir_path, decoder_path

    def _first_encoder(
        self, core: object, model: object, cache_dir: Path, npu_problem: str | None,
    ) -> "CompiledEncoder":
        """The first length on the NPU, or the CPU's any-length encoder when there is no NPU or it refuses."""
        if npu_problem is not None:
            _logger.warning("No working NPU: %s", npu_problem)
            self._note = "No working NPU found, so the CPU runs the model."
            return self._cpu_encoder(core, model, cache_dir)
        first = FIRST_BUCKET_FRAMES
        try:
            # Checked on silence just after it goes live, which keeps the check out of the start-up second.
            return CompiledEncoder({first: compile_length(core, model, cache_dir, _NPU, first, check=False)})
        except RuntimeError as refusal:
            return self._cpu_after_refusal(core, model, cache_dir, refusal)

    def _cpu_after_refusal(
        self, core: object, model: object, cache_dir: Path, refusal: Exception,
    ) -> "CompiledEncoder":
        _logger.warning("OpenVINO could not run the encoder on the NPU: %s", refusal)
        self._note = f"The NPU could not run the model ({refusal}), so the CPU took over."
        return self._cpu_encoder(core, model, cache_dir)

    def _cpu_encoder(self, core: object, model: object, cache_dir: Path) -> "CompiledEncoder":
        return CompiledEncoder({0: compile_length(core, model, cache_dir, _CPU, 0)})

    def _passes_check_on_silence(self, npu: "_NpuParakeet", encoder: "CompiledEncoder") -> bool:
        """Whether the NPU's first length gives finite output; when it doesn't, its CPU twin takes over."""
        try:
            with self._turn:
                encoder.check_on_silence()
        except RuntimeError as overflow:
            core, model, cache_dir = self._cpu_parts
            cpu = npu.with_encoder(self._cpu_after_refusal(core, model, cache_dir, overflow))
            self._go_live(cpu, every_length=True)
            return False
        return True

    def _add_other_lengths(
        self, encoder: "CompiledEncoder", core: object, model: object, cache_dir: Path, *, cold: bool,
    ) -> None:
        rest = [frames for frames in BUCKET_FRAMES if frames != FIRST_BUCKET_FRAMES]
        with ThreadPoolExecutor(1 if cold else len(rest)) as pool:
            pending = {pool.submit(compile_length, core, model, cache_dir, _NPU, frames): frames for frames in rest}
            for future in as_completed(pending):
                _add_length(encoder, pending[future], future)

    def _go_live(self, npu: "_NpuParakeet", *, every_length: bool) -> None:
        """Hand dictation to *npu*; a stand-in keeps it until every length is up."""
        with self._turn:
            if self._disposed or (self._stand_in is not None and not every_length):
                return
            self._model, stand_in, self._stand_in = npu, self._stand_in, None
            if self._blas_threads_before is None:
                self._blas_threads_before = set_numpy_blas_threads(1)
        self._announce()
        if stand_in is not None:
            stand_in.dispose()
            _logger.info("The %s took over dictation from the CPU stand-in", npu.encoder_device)

    def _record_failure(self, failure: Exception) -> None:
        _logger.warning("The NPU engine could not load: %s", failure, exc_info=True)
        self._failure = failure
        if not self.is_model_loaded() or self._stand_in is not None:
            self._note = f"The NPU engine could not load ({failure}), so the CPU took over."

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
        return self._model is not None or self._stand_in is not None

    @property
    def encoder_device(self) -> str | None:
        """Where dictation runs: "NPU" or "CPU", as OpenVINO reports it; None before anything is up."""
        if self._stand_in is not None:
            return _CPU
        model = self._model
        return None if model is None else model.encoder_device

    @property
    def lengths_up(self) -> int:
        """How many fixed lengths the encoder has up; 0 while the CPU stand-in has dictation."""
        model = self._model
        return 0 if model is None or self._stand_in is not None else model.length_count

    @property
    def device_note(self) -> str | None:
        """Why dictation is not on the NPU, when it isn't."""
        if self._stand_in is not None:
            return self._note or "Preparing the NPU for its first start, so the CPU takes dictation meanwhile."
        return self._note

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        with self._turn:
            if self._stand_in is not None:
                return self._stand_in.transcribe(audio)
            if self._model is None:
                raise TranscriptionError("Model not loaded")
            start = time.monotonic()
            try:
                text = _text_of(self._model, audio)
            except RuntimeError as failure:
                text = self._text_after_npu_failure(audio, failure)
            return TranscriptionResult(
                text=text,
                language="en",
                language_probability=0.0,
                transcription_seconds=time.monotonic() - start,
            )

    def _text_after_npu_failure(self, audio: AudioSegment, failure: RuntimeError) -> str:
        """Move to the CPU when the NPU stops working mid-session, and transcribe *audio* there."""
        if self._model.encoder_device != _NPU or self._cpu_parts is None:
            raise failure
        _logger.warning("The NPU stopped working, moving to the CPU: %s", failure)
        self._note = f"The NPU stopped working ({failure}), so the CPU took over."
        core, model, cache_dir = self._cpu_parts
        self._model = self._model.with_encoder(self._cpu_encoder(core, model, cache_dir))
        return _text_of(self._model, audio)

    def dispose(self) -> None:
        with self._turn:
            self._disposed = True
            self._model, stand_in, self._stand_in = None, self._stand_in, None
            if self._blas_threads_before is not None:
                set_numpy_blas_threads(self._blas_threads_before)
                self._blas_threads_before = None
        if stand_in is not None:
            stand_in.dispose()


class CompiledEncoder:
    """The encoder compiled for one device: per padded length, or for any length (key 0)."""

    def __init__(self, compiled_by_frames: dict[int, object]) -> None:
        self._compiled_by_frames = compiled_by_frames
        # OpenVINO's own account of where the graph runs, not just the device it was asked for. The NPU
        # plugin answers "NPU"; the CPU and GPU plugins answer a list.
        devices = next(iter(compiled_by_frames.values())).get_property("EXECUTION_DEVICES")
        self.device = devices if isinstance(devices, str) else ",".join(devices)

    @property
    def length_count(self) -> int:
        return len(self._compiled_by_frames)

    @property
    def longest_samples(self) -> int:
        """The most audio one pass takes: the longest length up so far."""
        longest = max(BUCKET_FRAMES) if 0 in self._compiled_by_frames else max(self._compiled_by_frames)
        return (longest - 1) * _HOP_SAMPLES

    def check_on_silence(self) -> None:
        """Raise RuntimeError when a length gives non-finite output, as a device computing in fp16 can."""
        for frames, compiled in self._compiled_by_frames.items():
            _check_on_silence(compiled, self.device, frames)

    def add(self, frames: int, compiled: object) -> None:
        # A new dict, not an update: a transcription may be reading the old one.
        self._compiled_by_frames = {**self._compiled_by_frames, frames: compiled}

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
        compiled_by_frames = self._compiled_by_frames
        if 0 in compiled_by_frames:
            return frames, compiled_by_frames[0]
        fitting = [size for size in compiled_by_frames if size >= frames]
        if not fitting:
            raise TranscriptionError(f"{frames} feature frames exceed the longest compiled length")
        return min(fitting), compiled_by_frames[min(fitting)]


class _NpuParakeet:
    def __init__(self, encoder: CompiledEncoder, decoder: object, tokenizer: object) -> None:
        self._encoder = encoder
        self._decoder = decoder
        self._tokenizer = tokenizer
        # The native decoder makes the features too, ten times faster than NumPy.
        self._features = getattr(decoder, "log_mel", log_mel_features)

    @property
    def encoder_device(self) -> str:
        return self._encoder.device

    @property
    def longest_samples(self) -> int:
        return self._encoder.longest_samples

    @property
    def length_count(self) -> int:
        return self._encoder.length_count

    def with_encoder(self, encoder: CompiledEncoder) -> "_NpuParakeet":
        return _NpuParakeet(encoder, self._decoder, self._tokenizer)

    def transcribe(self, samples: np.ndarray) -> str:
        if samples.size <= _SHORTEST_SAMPLES:
            return ""
        encoded = self._encoder(self._features(samples))
        return str(self._tokenizer.DecodeIds(self._decoder.decode(encoded))).strip()


def _text_of(model: _NpuParakeet, audio: AudioSegment) -> str:
    pieces = split_at_pauses(audio.data, model.longest_samples)
    return " ".join(part for part in map(model.transcribe, pieces) if part)


def _load_decoder(openvino: object, decoder_path: Path) -> object:
    """The native decoder (``native/tdt``), 2.5x faster on a Core 5 320, or OpenVINO's when it isn't built."""
    with np.load(decoder_path) as loaded:
        tensors = dict(loaded)
    library = native_tdt_dll()
    if library is not None:
        try:
            return _native_decoder(library, tensors)
        except OSError:
            _logger.warning("The native TDT decoder could not load; decoding through OpenVINO", exc_info=True)
    return OpenVinoTdtDecoder(openvino, tensors)


def _native_decoder(library: Path, tensors: dict[str, np.ndarray]) -> object:
    from skryb.transcription.native_tdt_decoder import NativeTdtDecoder

    decoder = NativeTdtDecoder(library, tensors)
    _logger.info("Decoding with the native TDT decoder (%s)", decoder.kernel)
    return decoder


def _add_length(encoder: CompiledEncoder, frames: int, compiled: Future) -> None:
    try:
        encoder.add(frames, compiled.result())
    except RuntimeError as refusal:
        # The lengths already up keep working; longer audio is cut to fit them.
        _logger.warning("The NPU could not run the %d-frame length: %s", frames, refusal)


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


def _delete_staged_leftovers(export_dir: Path) -> None:
    """Files a conversion cut short staged (``encoder-int8.partial.bin``, 0.6 GB); logged, never raised."""
    for leftover in export_dir.glob("*.partial.*"):
        try:
            leftover.unlink()
        except OSError:
            _logger.warning("Could not delete %s", leftover, exc_info=True)


def _delete_read_only_tree(directory: Path) -> None:
    if not directory.is_dir():
        return
    # Adds the write bit only: a directory that lost its execute bit could no longer be emptied.
    for path in directory.rglob("*"):
        path.chmod(path.stat().st_mode | stat.S_IWRITE)
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


def compile_length(
    core: object, model: object, cache_dir: Path, device: str, frames: int, *, check: bool = True,
) -> object:
    """*model* compiled for *device* at *frames* feature frames (0: any length), checked on silence.

    The warm-up must come back finite: a device that computes in fp16 and
    overflows is caught here, and the caller moves to the CPU. It also
    takes the length's first-run cost off the first dictation.
    """
    shaped = model.clone()
    if frames:
        shaped.reshape({"features": [1, _N_MELS, frames], "length": [1]})
    config = {
        "CACHE_DIR": str(cache_dir),
        "PERFORMANCE_HINT": "LATENCY",
        **(_NPU_CONFIG if device == _NPU else {}),
    }
    compiled = core.compile_model(shaped, device, config)
    if check:
        _check_on_silence(compiled, device, frames)
    return compiled


def _check_on_silence(compiled: object, device: str, frames: int) -> None:
    probe = frames or _PROBE_FRAMES
    warm = compiled([np.zeros((1, _N_MELS, probe), np.float32), np.array([probe], np.int64)])[0]
    if not np.isfinite(warm).all():
        raise RuntimeError(f"the encoder gave non-finite values on {device}")


def _wake_npu(openvino: object, core: object, cache_dir: Path) -> str | None:
    """Start the NPU's driver with a one-op model; why the NPU can't be used, or None when it can."""
    ops = opset(openvino)
    x = ops.parameter([1, 16], np.float32, name="x")
    tiny = openvino.Model([ops.relu(x)], [x], "wake")
    try:
        core.compile_model(tiny, _NPU, {"CACHE_DIR": str(cache_dir)})([np.zeros((1, 16), np.float32)])
    except RuntimeError as refusal:
        return str(refusal)
    return None
