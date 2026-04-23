"""Inspect the HuggingFace model cache from tests.

Tests that need a real Whisper model check the cache before committing
to a load, so a fresh checkout never triggers a multi-gigabyte download
or a hard failure.  Discovery works by replicating
:meth:`HuggingFaceModelManager._snapshot_has_complete_model` against the
on-disk layout, without pulling in the download-side dependencies.
"""

from pathlib import Path

import pytest

from voicetotext.core.paths import AppPaths
from voicetotext.transcription.hf_model_manager import HuggingFaceModelManager
from voicetotext.transcription.model_info import TRANSFORMERS_WHISPER_MODELS


def models_dir() -> Path | None:
    try:
        return AppPaths().models_dir
    except KeyError:
        return None


def installed_model_names() -> list[str]:
    directory = models_dir()
    if directory is None or not directory.exists():
        return []
    installed: list[str] = []
    for name, info in TRANSFORMERS_WHISPER_MODELS.items():
        repo_dir = directory / f"models--{info.repo_id.replace('/', '--')}"
        snapshots = repo_dir / "snapshots"
        if not snapshots.exists():
            continue
        for snapshot in snapshots.iterdir():
            if HuggingFaceModelManager._snapshot_has_complete_model(snapshot):
                installed.append(name)
                break
    return installed


def require_installed_model(name: str) -> str:
    if name not in installed_model_names():
        pytest.skip(f"whisper model not installed locally: {name}")
    return name
