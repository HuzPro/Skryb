"""Tests for :class:`TransformersWhisperAdapter`.

Everything is driven through the public ``load_model`` / ``transcribe`` /
``dispose`` port with ``transformers`` and ``torch`` patched at the
module boundary — these are legitimate seams, not internal details.
A single helper hides the mock-setup boilerplate so each test reads
as a behavioural specification.
"""

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


def _mocked_modules() -> dict:
    return {"transformers": MagicMock(), "torch": MagicMock()}


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
    )


def _setup_transcribe_mocks(decoded_text: str = "") -> tuple:
    """Return (processor, model, features) configured so the adapter's
    ``transcribe`` path executes cleanly and produces ``decoded_text``."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained.return_value
    model = WhisperForConditionalGeneration.from_pretrained.return_value
    model.dtype = np.float32

    features = MagicMock()
    features.input_features.to.return_value = MagicMock()
    features.attention_mask.to.return_value = MagicMock()
    processor.return_value = features
    model.generate.return_value = MagicMock()
    processor.batch_decode.return_value = [decoded_text]

    return processor, model, features


class TestTransformersWhisperAdapterLoadModel:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_model_and_processor_from_hf_repo(self):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")

        adapter.load_model()

        WhisperProcessor.from_pretrained.assert_called_once_with(
            "openai/whisper-base"
        )
        WhisperForConditionalGeneration.from_pretrained.assert_called_once_with(
            "openai/whisper-base", dtype=torch.float32
        )
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_from_local_cache_when_download_root_set(self):
        import torch
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
            dtype=torch.float32,
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_force_fp32_dtype_when_loading_on_cpu(self):
        """Transformers 5.x honors ``torch_dtype`` from config.json by default,
        so large-v3-turbo (config: float16) lands on CPU as fp16 and cannot
        run Linear/Conv layers. The adapter must force fp32 on CPU."""
        import torch
        from transformers import WhisperForConditionalGeneration

        adapter = TransformersWhisperAdapter(model_size="large-v3-turbo", device="cpu")

        adapter.load_model()

        call_kwargs = WhisperForConditionalGeneration.from_pretrained.call_args[1]
        assert call_kwargs.get("dtype") is torch.float32

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_not_force_dtype_when_loading_on_cuda(self):
        """CUDA path relies on ``_optimize_for_device`` to cast to fp16 and
        must leave the HF default dtype selection alone."""
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        adapter = TransformersWhisperAdapter(model_size="large-v3-turbo", device="cuda")

        adapter.load_model()

        call_kwargs = WhisperForConditionalGeneration.from_pretrained.call_args[1]
        assert "dtype" not in call_kwargs

    def test_should_report_not_loaded_before_load(self):
        adapter = TransformersWhisperAdapter(model_size="base")

        assert adapter.is_model_loaded() is False


class TestTransformersWhisperAdapterTranscribe:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_return_decoded_text_in_transcription_result(self):
        _setup_transcribe_mocks(decoded_text=" hello world ")
        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()

        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.duration_seconds >= 0.0

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_raise_when_transcribing_before_load(self):
        adapter = TransformersWhisperAdapter(model_size="base")

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_language_and_beam_size_to_generate(self):
        _, model, _ = _setup_transcribe_mocks()
        adapter = TransformersWhisperAdapter(
            model_size="base", device="cpu", language="en", beam_size=3,
        )
        adapter.load_model()

        adapter.transcribe(_make_audio())

        kwargs = model.generate.call_args[1]
        assert kwargs["num_beams"] == 3
        assert kwargs["language"] == "en"
        assert kwargs["task"] == "transcribe"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_preprocess_audio_without_truncation(self):
        processor, _, _ = _setup_transcribe_mocks()
        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()

        adapter.transcribe(_make_audio())

        kwargs = processor.call_args[1]
        assert kwargs["truncation"] is False
        assert kwargs["padding"] == "longest"
        assert kwargs["return_attention_mask"] is True

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_attention_mask_through_to_generate(self):
        _, model, features = _setup_transcribe_mocks()
        attention_mask = features.attention_mask.to.return_value
        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()

        adapter.transcribe(_make_audio())

        kwargs = model.generate.call_args[1]
        assert kwargs["attention_mask"] is attention_mask

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_move_attention_mask_to_device(self):
        _, _, features = _setup_transcribe_mocks()
        adapter = TransformersWhisperAdapter(model_size="base", device="cuda")
        adapter.load_model()

        adapter.transcribe(_make_audio())

        features.attention_mask.to.assert_called_with("cuda")

    @pytest.mark.parametrize(
        ("kwarg", "expected"),
        [
            ("return_timestamps", True),
            ("temperature", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
            ("compression_ratio_threshold", 1.35),
            ("logprob_threshold", -1.0),
            ("no_speech_threshold", 0.6),
            ("condition_on_prev_tokens", True),
        ],
    )
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_whisper_decoding_defaults_to_generate(
        self, kwarg: str, expected: object
    ):
        _, model, _ = _setup_transcribe_mocks()
        adapter = TransformersWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()

        adapter.transcribe(_make_audio())

        kwargs = model.generate.call_args[1]
        assert kwargs[kwarg] == expected


class TestTransformersWhisperAdapterDispose:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_clear_model_on_dispose(self):
        adapter = TransformersWhisperAdapter(model_size="base")
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_release_gpu_cache_on_dispose(self):
        import torch

        adapter = TransformersWhisperAdapter(model_size="base", device="cuda")
        adapter.load_model()

        adapter.dispose()

        torch.cuda.empty_cache.assert_called_once()

    @patch.dict("sys.modules", _mocked_modules())
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
