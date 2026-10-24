import pytest

from skryb.transcription.result import TranscriptionResult


class TestTranscriptionResultCreation:
    def test_should_store_transcribed_text(self):
        result = TranscriptionResult(
            text="hello world",
            language="en",
            language_probability=0.95,
            transcription_seconds=1.5,
        )

        assert result.text == "hello world"
        assert result.language == "en"
        assert result.language_probability == 0.95
        assert result.transcription_seconds == 1.5

    def test_should_detect_empty_text(self):
        result = TranscriptionResult(
            text="",
            language="en",
            language_probability=0.0,
            transcription_seconds=0.0,
        )

        assert result.is_empty is True

    def test_should_detect_whitespace_only_as_empty(self):
        result = TranscriptionResult(
            text="   \n\t  ",
            language="en",
            language_probability=0.0,
            transcription_seconds=0.0,
        )

        assert result.is_empty is True

    def test_should_detect_non_empty_text(self):
        result = TranscriptionResult(
            text="some words",
            language="en",
            language_probability=0.9,
            transcription_seconds=2.0,
        )

        assert result.is_empty is False


class TestTranscriptionResultImmutability:
    def test_should_be_immutable(self):
        result = TranscriptionResult(
            text="hello",
            language="en",
            language_probability=0.9,
            transcription_seconds=1.0,
        )

        with pytest.raises(AttributeError):
            result.text = "changed"
