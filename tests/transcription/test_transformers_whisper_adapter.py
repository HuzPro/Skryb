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

        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")

        adapter.load_model()

        WhisperProcessor.from_pretrained.assert_called_once_with("openai/whisper-base")
        WhisperForConditionalGeneration.from_pretrained.assert_called_once_with(
            "openai/whisper-base"
        )
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_pass_cache_dir_and_offline_when_download_root_set(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        adapter = TransformersWhisperAdapter(
            model_size="small", device="cpu", download_root="/tmp/models",
        )

        adapter.load_model()

        WhisperProcessor.from_pretrained.assert_called_once_with(
            "openai/whisper-small",
            cache_dir="/tmp/models",
            local_files_only=True,
        )
        WhisperForConditionalGeneration.from_pretrained.assert_called_once_with(
            "openai/whisper-small",
            cache_dir="/tmp/models",
            local_files_only=True,
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
        mock_features.attention_mask.to.return_value = MagicMock()
        mock_processor.return_value = mock_features

        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = [" hello world "]

        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()
        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.duration_seconds >= 0.0

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
        mock_features.attention_mask.to.return_value = MagicMock()
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

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_call_processor_without_truncation(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        mock_processor = WhisperProcessor.from_pretrained.return_value
        mock_model = WhisperForConditionalGeneration.from_pretrained.return_value
        mock_model.dtype = np.float32

        mock_features = MagicMock()
        mock_features.input_features.to.return_value = MagicMock()
        mock_features.attention_mask.to.return_value = MagicMock()
        mock_processor.return_value = mock_features
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = [""]

        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()
        adapter.transcribe(_make_audio())

        call_kwargs = mock_processor.call_args[1]
        assert call_kwargs["truncation"] is False
        assert call_kwargs["padding"] == "longest"
        assert call_kwargs["return_attention_mask"] is True

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_pass_attention_mask_to_generate(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        mock_processor = WhisperProcessor.from_pretrained.return_value
        mock_model = WhisperForConditionalGeneration.from_pretrained.return_value
        mock_model.dtype = np.float32

        mock_features = MagicMock()
        mock_features.input_features.to.return_value = MagicMock()
        mock_attention_mask = MagicMock()
        mock_features.attention_mask.to.return_value = mock_attention_mask
        mock_processor.return_value = mock_features
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = [""]

        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()
        adapter.transcribe(_make_audio())

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["attention_mask"] is mock_attention_mask

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_pass_return_timestamps_to_generate(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        mock_processor = WhisperProcessor.from_pretrained.return_value
        mock_model = WhisperForConditionalGeneration.from_pretrained.return_value
        mock_model.dtype = np.float32

        mock_features = MagicMock()
        mock_features.input_features.to.return_value = MagicMock()
        mock_features.attention_mask.to.return_value = MagicMock()
        mock_processor.return_value = mock_features
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = [""]

        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()
        adapter.transcribe(_make_audio())

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["return_timestamps"] is True

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_move_attention_mask_to_device(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        mock_processor = WhisperProcessor.from_pretrained.return_value
        mock_model = WhisperForConditionalGeneration.from_pretrained.return_value
        mock_model.dtype = np.float32

        mock_features = MagicMock()
        mock_features.input_features.to.return_value = MagicMock()
        mock_features.attention_mask.to.return_value = MagicMock()
        mock_processor.return_value = mock_features
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = [""]

        adapter = TransformersWhisperAdapter(model_size="base", device="cuda")
        adapter.load_model()
        adapter.transcribe(_make_audio())

        mock_features.attention_mask.to.assert_called_with("cuda")

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_pass_temperature_fallback_to_generate(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        mock_processor = WhisperProcessor.from_pretrained.return_value
        mock_model = WhisperForConditionalGeneration.from_pretrained.return_value
        mock_model.dtype = np.float32

        mock_features = MagicMock()
        mock_features.input_features.to.return_value = MagicMock()
        mock_features.attention_mask.to.return_value = MagicMock()
        mock_processor.return_value = mock_features
        mock_model.generate.return_value = MagicMock()
        mock_processor.batch_decode.return_value = [""]

        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()
        adapter.transcribe(_make_audio())

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["temperature"] == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class TestTransformersWhisperAdapterDispose:
    @patch.dict("sys.modules", _mock_transformers())
    def test_should_clear_model_on_dispose(self):
        adapter = TransformersWhisperAdapter(model_size="base")
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_release_gpu_cache_on_dispose(self):
        import torch

        adapter = TransformersWhisperAdapter(model_size="base", device="cuda")
        adapter.load_model()

        adapter.dispose()

        torch.cuda.empty_cache.assert_called_once()

    @patch.dict("sys.modules", _mock_transformers())
    def test_should_skip_gpu_cache_release_on_cpu_dispose(self):
        import torch

        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()

        adapter.dispose()

        torch.cuda.empty_cache.assert_not_called()

    def test_should_not_raise_on_dispose_when_not_loaded(self):
        adapter = TransformersWhisperAdapter(model_size="base")

        adapter.dispose()


class TestTransformersWhisperAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self):
        adapter = TransformersWhisperAdapter(model_size="base")

        assert isinstance(adapter, TranscriptionEngine)
