"""One-time build of a CTranslate2 copy of a downloaded Whisper snapshot.

The CTranslate2 runtime cannot read HuggingFace safetensors; it needs its
own ``model.bin`` plus ``vocabulary.json``. Producing those takes ~9 s of
CPU work and imports torch and transformers, so it happens once when the
weights are installed rather than on any transcription path.

The build lands *inside* the snapshot directory it was produced from
(``snapshots/<sha>/.ct2``). That is what keys it to the source revision
for free: the pinned commit is already in the path, so changing
``ModelInfo.revision`` names a different snapshot, which has no ``.ct2``
directory and is therefore rebuilt. Deleting the model
(``HuggingFaceModelManager.delete`` removes the whole ``models--*`` tree)
takes the build with it, and the ``refs/main`` completion barrier the
install check relies on is untouched - a subdirectory is invisible to
both ``_is_nemo_layout`` and ``_is_safetensors_layout``.

The build is staged under a sibling name and renamed into place as the
last step, mirroring the ``.part`` -> target rename the file downloader
already uses. A crash or a failed conversion therefore leaves staging
debris that the next attempt discards, never a half-written ``.ct2`` that
looks finished.
"""

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

from skryb.transcription.backend_imports import (
    make_engine_dependency_error,
)


_logger = logging.getLogger(__name__)

# Dot-prefixed so it can never collide with a directory the upstream repo
# ships at the pinned commit.
_CT2_DIRECTORY_NAME = ".ct2"
_STAGING_DIRECTORY_NAME = ".ct2.incomplete"

# The checkpoint is bfloat16 on disk, so converting at bfloat16 stores the
# values unchanged. Verified token-identical to the transformers engine in
# both decoder-prefix modes.
_QUANTIZATION = "bfloat16"

# CTranslate2 writes these three itself and raises on a name collision, so
# they are never copied across from the source snapshot.
_REQUIRED_CT2_FILENAMES = ("model.bin", "config.json", "vocabulary.json")

# The tokenizer and feature-extractor assets a CTranslate2 engine still
# needs; whichever of them the snapshot actually holds get copied.
_COPIED_ASSET_NAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
)

_WHISPER_MODEL_TYPE = "whisper"

SnapshotConverter = Callable[[Path, Path], None]


def ct2_model_dir(snapshot: Path) -> Path:
    """Where the CTranslate2 build of *snapshot* lives."""
    return snapshot / _CT2_DIRECTORY_NAME


def snapshot_has_ct2_model(snapshot: Path) -> bool:
    """Whether *snapshot* holds a CTranslate2 build the runtime can open."""
    built = ct2_model_dir(snapshot)
    return all((built / name).is_file() for name in _REQUIRED_CT2_FILENAMES)


def convert_snapshot(snapshot: Path, output: Path) -> None:
    """Run CTranslate2's transformers converter over *snapshot*."""
    converter_type = _import_converter()
    converter = converter_type(
        str(snapshot), copy_files=list(_copyable_asset_names(snapshot)),
    )
    converter.convert(str(output), quantization=_QUANTIZATION)


def ensure_ct2_model(
    snapshot: Path, convert: SnapshotConverter = convert_snapshot,
) -> Path:
    """Return the CTranslate2 build of *snapshot*, building it if absent.

    Raises ``EngineDependencyMissingError`` when the CTranslate2 package
    is not installed, and whatever the converter raises when the snapshot
    cannot be converted. An engine load wants that failure loud; the
    post-install path uses :func:`prebuild_ct2_model` instead.
    """
    if snapshot_has_ct2_model(snapshot):
        return ct2_model_dir(snapshot)
    return _build_ct2_model(snapshot, convert)


def prebuild_ct2_model(
    snapshot: Path, convert: SnapshotConverter = convert_snapshot,
) -> None:
    """Build the CTranslate2 copy at install time, best effort.

    Runs as the download job's post-install step so the cost lands where
    the user is already watching a progress bar. A failure here is not a
    failed download: the safetensors weights are installed and the
    transformers engine can open them, and :func:`ensure_ct2_model`
    retries the build the next time a CTranslate2 engine loads.
    """
    if not _is_whisper_snapshot(snapshot):
        return
    # ponytail: converts in-process, so a CTranslate2 wheel built for the
    # wrong runtime aborts the app with OMP: Error #15 instead of raising.
    # Move to a subprocess if mismatched wheels turn out to be common.
    try:
        ensure_ct2_model(snapshot, convert)
    except Exception as exc:
        _logger.warning(
            "CTranslate2 build for '%s' failed (%s); it will be retried "
            "on the next engine load.", snapshot, exc,
        )


def _build_ct2_model(snapshot: Path, convert: SnapshotConverter) -> Path:
    staging = snapshot / _STAGING_DIRECTORY_NAME
    built = ct2_model_dir(snapshot)
    _discard(staging)
    _discard(built)
    try:
        convert(snapshot, staging)
    except Exception:
        _discard(staging)
        raise
    staging.replace(built)
    return built


def _discard(directory: Path) -> None:
    shutil.rmtree(directory, ignore_errors=True)


def _copyable_asset_names(snapshot: Path) -> tuple[str, ...]:
    return tuple(
        name for name in _COPIED_ASSET_NAMES if (snapshot / name).is_file()
    )


def _is_whisper_snapshot(snapshot: Path) -> bool:
    """Whether *snapshot* is a model CTranslate2's Whisper loader accepts.

    Read from the snapshot rather than from the catalogue so the download
    job stays ignorant of which catalogue entries have a CTranslate2
    engine.
    """
    return _declared_model_type(snapshot / "config.json") == _WHISPER_MODEL_TYPE


def _declared_model_type(config_path: Path) -> str:
    try:
        config: object = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
    if not isinstance(config, dict):
        return ""
    model_type: object = cast(dict[object, object], config).get("model_type")
    return model_type if isinstance(model_type, str) else ""


def _import_converter() -> type:
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError as exc:
        raise make_engine_dependency_error(
            package_name="ctranslate2",
            install_hint=(
                "CTranslate2 not installed: see INSTALL.md for the "
                "ROCm / CUDA / CPU build to use"
            ),
            underlying=exc,
        ) from exc
    return TransformersConverter
