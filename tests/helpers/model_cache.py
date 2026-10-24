"""Locate the application model cache for opt-in integration tests."""

from pathlib import Path

from skryb.core.paths import AppPaths


def models_dir() -> Path | None:
    try:
        return AppPaths().models_dir
    except KeyError:
        return None
