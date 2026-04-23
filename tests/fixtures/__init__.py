"""Audio-fixture helpers for tests that need real MP3 input.

The actual audio files live under ``tests/fixtures/audio/`` and are
**not** checked into git (they are large binaries).  Tests that depend
on them call ``require_short_fixture()`` / ``require_long_fixture()``
to get a path, which skips the test with a clear message when the file
is absent.
"""

from pathlib import Path

import pytest


FIXTURE_DIR = Path(__file__).parent / "audio"
SHORT_MP3 = FIXTURE_DIR / "Testing_1_2_3.mp3"
LONG_MP3 = FIXTURE_DIR / "LongStory.mp3"


def require_short_fixture() -> Path:
    if not SHORT_MP3.exists():
        pytest.skip(f"audio fixture missing: {SHORT_MP3}")
    return SHORT_MP3


def require_long_fixture() -> Path:
    if not LONG_MP3.exists():
        pytest.skip(f"audio fixture missing: {LONG_MP3}")
    return LONG_MP3
