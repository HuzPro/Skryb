"""Phonon (2 and 2.5 English) on the CPU, through Fermion Research's own CPU engine.

Phonon is parakeet-tdt-0.6b-v3 with its encoder stored at about 2.1 bits
per weight. Fermion's engine (the pinned ``fermion-research`` package) runs
the matrix maths on those packed weights with native AVX2 / AVX-512 / NEON
kernels; expanding them back to full size, as nano-parakeet would, costs
as much as Parakeet v3 itself.

The repo ships the model as one zstd tar, named with the profile Fermion's
engine needs in the repo's ``config.json`` ("artifact"). The engine loads the
unpacked directory, so the archive is unpacked once, inside the snapshot
it came from (``snapshots/<sha>/.phonon2``), the way the CTranslate2 build
is keyed to its snapshot. Fermion's unpacker verifies every member's
SHA-256, stages under a sibling name and renames into place last, so an
interrupted unpack never leaves a directory that looks finished.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from skryb.core.exceptions import TranscriptionError
from skryb.recording.audio_segment import AudioSegment
from skryb.transcription.backend_imports import (
    make_engine_dependency_error,
)
from skryb.transcription.engine import TranscriptionEngine
from skryb.transcription.result import TranscriptionResult


_logger = logging.getLogger(__name__)

# Dot-prefixed so it can never collide with a directory the upstream repo
# ships at the pinned commit.
_UNPACKED_DIRECTORY_NAME = ".phonon2"
_CONTAINER_FILENAME = "model.fermion"
_THREADS_ENV = "FERMION_CPU_THREADS"
_MAX_THREADS = 16
_INSTALL_HINT = (
    "Phonon needs Fermion's CPU engine. Install it with: "
    "pip install fermion-research==0.2.15"
)


def unpacked_model_dir(snapshot: Path) -> Path:
    """Where the unpacked Phonon model for *snapshot* lives."""
    return snapshot / _UNPACKED_DIRECTORY_NAME


class Phonon2CpuAdapter(TranscriptionEngine):
    """Phonon on Fermion's CPU engine. It never uses the GPU."""

    def __init__(self, snapshot: Path) -> None:
        self._snapshot = snapshot
        self._model: object | None = None

    def load_model(self) -> None:
        started_at = time.monotonic()
        engine, fetch = _import_backend()
        artifact = read_artifact(self._snapshot)
        model_dir = _ensure_unpacked(fetch, self._snapshot, artifact["filename"])
        _set_thread_count_without_powershell()
        try:
            self._model = _load_model_from_disk(engine, model_dir, artifact)
        except SystemExit as refusal:
            # Fermion refuses with SystemExit, which would otherwise end the
            # engine-load thread without a word.
            raise TranscriptionError(f"Phonon could not load: {refusal}") from refusal
        _logger.info(
            "[load_model phonon] %s complete in %.2fs on %s threads", artifact["profile"],
            time.monotonic() - started_at, os.environ.get(_THREADS_ENV, "default"),
        )

    def is_model_loaded(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: AudioSegment) -> TranscriptionResult:
        if self._model is None:
            raise TranscriptionError("Model not loaded")
        start = time.monotonic()
        # Fermion cuts audio over 35 s into windows at pauses itself.
        text, _decode_seconds, _audio_seconds = self._model.transcribe_array(audio.data)
        return TranscriptionResult(
            text=str(text).strip(),
            language="en",
            language_probability=0.0,
            transcription_seconds=time.monotonic() - start,
        )

    def dispose(self) -> None:
        self._model = None


def _import_backend() -> tuple[object, object]:
    try:
        return _import_fermion_modules()
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="fermion", install_hint=_INSTALL_HINT, underlying=exc,
        ) from exc


def _import_fermion_modules() -> tuple[object, object]:
    from fermion._speech import engine_phonon2_cpu, fetch

    return engine_phonon2_cpu, fetch


def read_artifact(snapshot: Path) -> dict:
    """The archive's file name and Fermion's profile and backend for it, from the repo's config.json."""
    try:
        artifact = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))["artifact"]
        return {key: str(artifact[key]) for key in ("filename", "profile", "backend")}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise TranscriptionError(f"{snapshot.name} has no usable Phonon config.json: {error}") from error


def _ensure_unpacked(fetch: object, snapshot: Path, archive: str) -> Path:
    model_dir = unpacked_model_dir(snapshot)
    if (model_dir / _CONTAINER_FILENAME).is_file():
        return model_dir
    _logger.info("Unpacking %s into %s", archive, model_dir)
    fetch._unpack(snapshot / archive, model_dir)
    return model_dir


def _load_model_from_disk(engine: object, model_dir: Path, artifact: dict) -> object:
    return engine.load(model_dir, profile=artifact["profile"], backend=artifact["backend"], quiet=True)


def _set_thread_count_without_powershell() -> None:
    """Pick Fermion's thread count for it, on Windows.

    Fermion counts physical cores by starting PowerShell, which flashes a
    console window under the windowless launcher on every load. This is
    the same rule it applies to the count: one thread per physical core
    from six cores up, every logical CPU below that, at most sixteen. A
    value already in the environment wins.
    """
    if sys.platform != "win32" or os.environ.get(_THREADS_ENV):
        return
    import psutil

    logical = os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    threads = physical if physical >= 6 else logical
    os.environ[_THREADS_ENV] = str(max(1, min(threads, _MAX_THREADS)))
