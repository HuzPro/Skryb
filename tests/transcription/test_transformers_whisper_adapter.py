from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult
from voicetotext.transcription.transformers_whisper_adapter import (
    TransformersWhisperAdapter,
)


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
    )


def _mock_transformers():
    """Set up mocked transformers module in sys.modules."""
    mock_transformers = MagicMock()
    return {
        "transformers": mock_transformers,
        "torch": MagicMock(),
    }


class TestTransformersWhisperAdapterLoadModel:
    @patch.dict("sys.modules", _mock_transformers())
    def test_should_load_model_with_given_parameters(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        adapter = TransformersWhisperAdapter(
            model_size="base", device="cpu",
        )

        adapter.load_model()

        WhisperProcessor.from_pretrained.assert_called_once_with("openai/whisper-base")
        WhisperForConditionalGeneration.from_pretrained.assert_called_once_with(
            "openai/whisper-base"
        )
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_pass_cache_dir_when_download_root_set(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        adapter = TransformersWhisperAdapter(
            model_size="small", device="cpu", download_root="/tmp/models",
        )

        adapter.load_model()

        WhisperProcessor.from_pretrained.assert_called_once_with(
            "openai/whisper-small", cache_dir="/tmp/models"
        )
        WhisperForConditionalGeneration.from_pretrained.assert_called_once_with(
            "openai/whisper-small", cache_dir="/tmp/models"
        )

    def test_should_report_not_loaded_before_load(self):
        adapter = TransformersWhisperAdapter(model_size="base")

        assert adapter.is_model_loaded() is False


class TestTransformersWhisperAdapterTranscribe:
    @patch.dict("sys.modules", _mock_transformers())
    def test_should_return_transcription_result(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        mock_processor = WhisperProcessor.from_pretrained.return_value
        mock_model = WhisperForConditionalGeneration.from_pretrained.return_value
        mock_model.dtype = np.float32

        mock_features = MagicMock()
        mock_features.input_features.to.return_value = MagicMock()
        mock_processor.return_value = mock_features

        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = [" hello world "]

        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()
        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.duration_seconds > 0

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_raise_when_model_not_loaded(self):
        adapter = TransformersWhisperAdapter(model_size="base")

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_pass_language_and_beam_size(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        mock_processor = WhisperProcessor.from_pretrained.return_value
        mock_model = WhisperForConditionalGeneration.from_pretrained.return_value
        mock_model.dtype = np.float32

        mock_features = MagicMock()
        mock_features.input_features.to.return_value = MagicMock()
        mock_processor.return_value = mock_features
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = [""]

        adapter = TransformersWhisperAdapter(
            model_size="base", device="cpu", language="en", beam_size=3,
        )
        adapter.load_model()
        adapter.transcribe(_make_audio())

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["num_beams"] == 3
        assert call_kwargs["language"] == "en"
        assert call_kwargs["task"] == "transcribe"


class TestTransformersWhisperAdapterDispose:
    @patch.dict("sys.modules", _mock_transformers())
    def test_should_clear_model_on_dispose(self):
        adapter = TransformersWhisperAdapter(model_size="base")
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    def test_should_not_raise_on_dispose_when_not_loaded(self):
        adapter = TransformersWhisperAdapter(model_size="base")

        adapter.dispose()


class TestTransformersWhisperAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self):
        adapter = TransformersWhisperAdapter(model_size="base")

        assert isinstance(adapter, TranscriptionEngine)
