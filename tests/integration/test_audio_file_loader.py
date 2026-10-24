"""Behaviour-focused tests for the MP3-to-``AudioSegment`` loader.

The loader is the single seam between a raw audio file on disk and the
``TranscriptionEngine`` port.  It must:

* fail fast on a missing path,
* return an ``AudioSegment`` whose shape, dtype, and sample-rate match
  what the engine expects downstream,
* cope with the fixture files we ship (mono 24 kHz MP3s).

Tests reach the real file system and decode real MP3 data, so they are
tagged ``integration``.  Skipping happens automatically when the
fixture files are absent, never as a hard failure.
"""

from pathlib import Path

import numpy as np
import pytest

from skryb.recording.audio_segment import AudioSegment

from tests.fixtures import require_long_fixture, require_short_fixture
from tests.helpers.audio_file_loader import load_audio_file


pytestmark = pytest.mark.integration


class TestAudioFileLoader:
    def test_should_raise_file_not_found_when_path_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does-not-exist.mp3"

        with pytest.raises(FileNotFoundError):
            load_audio_file(missing)

    def test_should_produce_audio_segment_when_loading_short_fixture(self) -> None:
        segment = load_audio_file(require_short_fixture())

        assert isinstance(segment, AudioSegment)

    def test_should_report_positive_duration_when_loading_short_fixture(
        self,
    ) -> None:
        segment = load_audio_file(require_short_fixture())

        assert segment.duration_seconds > 0.0

    def test_should_produce_float32_samples_when_loading_short_fixture(
        self,
    ) -> None:
        segment = load_audio_file(require_short_fixture())

        assert segment.data.dtype == np.float32

    def test_should_resample_to_sixteen_kilohertz_when_loading_short_fixture(
        self,
    ) -> None:
        segment = load_audio_file(require_short_fixture())

        assert segment.sample_rate == 16000

    def test_should_resample_to_sixteen_kilohertz_when_loading_long_fixture(
        self,
    ) -> None:
        segment = load_audio_file(require_long_fixture())

        assert segment.sample_rate == 16000

    def test_should_produce_one_dimensional_mono_array_when_loading_short_fixture(
        self,
    ) -> None:
        segment = load_audio_file(require_short_fixture())

        assert segment.data.ndim == 1

    def test_should_report_duration_over_two_minutes_when_loading_long_story_fixture(
        self,
    ) -> None:
        segment = load_audio_file(require_long_fixture())

        assert segment.duration_seconds > 120.0
