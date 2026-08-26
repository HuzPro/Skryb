import importlib
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import requests

PUNCTUATION_MODEL_NAME = "sherpa-onnx-online-punct-en-2024-08-06"
PUNCTUATION_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "punctuation-models/"
    "sherpa-onnx-online-punct-en-2024-08-06.tar.bz2"
)

_MODEL_FILENAME = "model.int8.onnx"
_VOCAB_FILENAME = "bpe.vocab"
_TERMINAL_PUNCTUATION = frozenset(".!?")
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_logger = logging.getLogger(__name__)
_warning_emitted = False


class PunctuationBackend(Protocol):
    def punctuate(self, text: str) -> str:
        ...


class _SherpaPunctuator(Protocol):
    def add_punctuation_with_case(self, text: str) -> str:
        ...


class _SherpaModule(Protocol):
    def OnlinePunctuationModelConfig(
        self,
        *,
        cnn_bilstm: str,
        bpe_vocab: str,
        num_threads: int,
        debug: bool,
        provider: str,
    ) -> object:
        ...

    def OnlinePunctuationConfig(self, *, model_config: object) -> object:
        ...

    def OnlinePunctuation(self, config: object) -> _SherpaPunctuator:
        ...


class PunctuationStage:
    def __init__(self, backend: PunctuationBackend | None) -> None:
        self._backend = backend

    def apply(self, text: str) -> str:
        if any(mark in text for mark in _TERMINAL_PUNCTUATION):
            return text
        if self._backend is None:
            return text
        try:
            return self._backend.punctuate(text)
        except (OSError, RuntimeError, ValueError) as exc:
            _warn_once("Punctuation restoration unavailable: %s", exc)
            return text


class SherpaOnnxPunctuationBackend:
    def __init__(self, punctuator: _SherpaPunctuator) -> None:
        self._punctuator = punctuator

    def punctuate(self, text: str) -> str:
        return self._punctuator.add_punctuation_with_case(text)


def sherpa_onnx_available() -> bool:
    return _import_sherpa_onnx() is not None


def create_punctuation_backend(models_dir: Path) -> PunctuationBackend | None:
    module = _import_sherpa_onnx()
    if module is None:
        _warn_once(
            "Punctuation restoration unavailable because sherpa-onnx is not "
            "installed; run 'pip install voicetotext[punct]'."
        )
        return None
    model_dir = models_dir / PUNCTUATION_MODEL_NAME
    if not _model_is_complete(model_dir) and not _fetch_model(models_dir):
        _warn_once("Punctuation restoration unavailable: model download failed.")
        return None
    try:
        model_config = module.OnlinePunctuationModelConfig(
            cnn_bilstm=str(model_dir / _MODEL_FILENAME),
            bpe_vocab=str(model_dir / _VOCAB_FILENAME),
            num_threads=1,
            debug=False,
            provider="cpu",
        )
        config = module.OnlinePunctuationConfig(model_config=model_config)
        punctuator = module.OnlinePunctuation(config)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _warn_once("Punctuation restoration unavailable: %s", exc)
        return None
    return SherpaOnnxPunctuationBackend(punctuator)


def _import_sherpa_onnx() -> _SherpaModule | None:
    try:
        module: ModuleType = importlib.import_module("sherpa_onnx")
    except ImportError:
        return None
    return cast(_SherpaModule, module)


def _model_is_complete(model_dir: Path) -> bool:
    return all(
        (model_dir / filename).is_file()
        for filename in (_MODEL_FILENAME, _VOCAB_FILENAME)
    )


def _fetch_model(models_dir: Path) -> bool:
    archive_path: Path | None = None
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        with requests.get(PUNCTUATION_MODEL_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(
                dir=models_dir,
                prefix=f"{PUNCTUATION_MODEL_NAME}-",
                suffix=".tar.bz2",
                delete=False,
            ) as archive_file:
                archive_path = Path(archive_file.name)
                for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        archive_file.write(chunk)
        _extract_model(archive_path, models_dir)
        return _model_is_complete(models_dir / PUNCTUATION_MODEL_NAME)
    except (OSError, requests.RequestException, tarfile.TarError, ValueError) as exc:
        _warn_once("Punctuation restoration unavailable: model fetch failed: %s", exc)
        return False
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)


def _extract_model(archive_path: Path, models_dir: Path) -> None:
    root = models_dir.resolve()
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        members = archive.getmembers()
        for member in members:
            destination = (models_dir / member.name).resolve()
            if os.path.commonpath((str(root), str(destination))) != str(root):
                raise ValueError("punctuation model archive contains an unsafe path")
            if member.issym() or member.islnk():
                raise ValueError("punctuation model archive contains a link")
        archive.extractall(models_dir, members=members)


def _warn_once(message: str, *args: object) -> None:
    global _warning_emitted
    if _warning_emitted:
        return
    _warning_emitted = True
    _logger.warning(message, *args)
