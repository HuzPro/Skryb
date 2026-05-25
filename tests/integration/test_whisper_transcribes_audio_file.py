"""Real Whisper transcribing real audio files.

Every test in this module runs once per Whisper model installed in the
user's HuggingFace cache — the ``real_whisper_engine`` fixture in
``conftest.py`` drives the parametrization.  When no models are
installed the fixture simply has no parameters and every test in this
module is skipped.

Assertions are deliberately lenient.  Whisper's output varies across
models and runs: "testing 1 2 3" might render as ``Testing, 1, 2, 3.``
or ``testing one two three``; long-form output can shift a word here
and there.  The tests pin down behaviours that *any* working Whisper
model should satisfy on clean TTS audio, not exact strings.
"""

import re

import pytest

from voicetotext.transcription.engine import TranscriptionEngine

from tests.fixtures import require_long_fixture, require_short_fixture
from tests.helpers.audio_file_loader import load_audio_file


pytestmark = [pytest.mark.integration, pytest.mark.slow]


_WORD_RE = re.compile(r"[a-z0-9]+")


def _bag_of_words(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class TestTranscribeShortAudio:
    def test_should_include_the_word_testing_when_transcribing_short_clean_audio(
        self, real_whisper_engine: TranscriptionEngine
    ) -> None:
        audio = load_audio_file(require_short_fixture())

        result = real_whisper_engine.transcribe(audio)

        assert "testing" in result.text.lower()

    def test_should_produce_non_empty_text_when_transcribing_short_audio(
        self, real_whisper_engine: TranscriptionEngine
    ) -> None:
        audio = load_audio_file(require_short_fixture())

        result = real_whisper_engine.transcribe(audio)

        assert result.text.strip() != ""

    def test_should_report_positive_inference_duration_when_transcribing_short_audio(
        self, real_whisper_engine: TranscriptionEngine
    ) -> None:
        audio = load_audio_file(require_short_fixture())

        result = real_whisper_engine.transcribe(audio)

        assert result.transcription_seconds > 0.0


class TestTranscribeLongAudio:
    def test_should_produce_non_empty_text_when_transcribing_long_story(
        self, real_whisper_engine: TranscriptionEngine
    ) -> None:
        audio = load_audio_file(require_long_fixture())

        result = real_whisper_engine.transcribe(audio)

        assert result.text.strip() != ""

    def test_should_produce_multi_word_text_when_transcribing_long_story(
        self, real_whisper_engine: TranscriptionEngine
    ) -> None:
        audio = load_audio_file(require_long_fixture())

        result = real_whisper_engine.transcribe(audio)

        word_count = len(_WORD_RE.findall(result.text))
        assert word_count > 100


class TestTranscribeIsRepeatable:
    def test_should_produce_similar_text_when_transcribing_same_short_audio_twice(
        self, real_whisper_engine: TranscriptionEngine
    ) -> None:
        audio = load_audio_file(require_short_fixture())

        first = real_whisper_engine.transcribe(audio)
        second = real_whisper_engine.transcribe(audio)

        similarity = _jaccard(_bag_of_words(first.text), _bag_of_words(second.text))
        assert similarity >= 0.9
