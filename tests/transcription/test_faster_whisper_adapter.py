import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.faster_whisper_adapter import FasterWhisperAdapter
from voicetotext.transcription.result import TranscriptionResult


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
    )


def _make_segment(text: str, start: float = 0.0, end: float = 1.0):
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    return seg


class TestFasterWhisperAdapterLoadModel:
    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_load_model_with_given_parameters(self, mock_whisper_cls):
        adapter = FasterWhisperAdapter(
            model_size="base",
            device="cpu",
            compute_type="int8",
        )

        adapter.load_model()

        mock_whisper_cls.assert_called_once_with(
            "base", device="cpu", compute_type="int8"
        )
        assert adapter.is_model_loaded() is True

    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_report_not_loaded_before_load(self, mock_whisper_cls):
        adapter = FasterWhisperAdapter(model_size="base")

        assert adapter.is_model_loaded() is False


class TestFasterWhisperAdapterTranscribe:
    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_return_transcription_result(self, mock_whisper_cls):
        mock_model = mock_whisper_cls.return_value
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95
        mock_info.duration = 2.0
        segments = [_make_segment(" hello"), _make_segment(" world")]
        mock_model.transcribe.return_value = (iter(segments), mock_info)

        adapter = FasterWhisperAdapter(model_size="base")
        adapter.load_model()
        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.language_probability == 0.95

    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_return_empty_result_for_no_segments(self, mock_whisper_cls):
        mock_model = mock_whisper_cls.return_value
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.0
        mock_info.duration = 0.5
        mock_model.transcribe.return_value = (iter([]), mock_info)

        adapter = FasterWhisperAdapter(model_size="base")
        adapter.load_model()
        result = adapter.transcribe(_make_audio())

        assert result.is_empty is True

    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_raise_when_model_not_loaded(self, mock_whisper_cls):
        adapter = FasterWhisperAdapter(model_size="base")

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())

    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_pass_language_and_beam_size(self, mock_whisper_cls):
        mock_model = mock_whisper_cls.return_value
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.9
        mock_info.duration = 1.0
        mock_model.transcribe.return_value = (iter([]), mock_info)

        adapter = FasterWhisperAdapter(
            model_size="base",
            language="en",
            beam_size=3,
            vad_filter=True,
        )
        adapter.load_model()
        adapter.transcribe(_make_audio())

        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"
        assert call_kwargs["beam_size"] == 3
        assert call_kwargs["vad_filter"] is True


class TestFasterWhisperAdapterDispose:
    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_clear_model_on_dispose(self, mock_whisper_cls):
        adapter = FasterWhisperAdapter(model_size="base")
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_not_raise_on_dispose_when_not_loaded(self, mock_whisper_cls):
        adapter = FasterWhisperAdapter(model_size="base")

        adapter.dispose()


class TestFasterWhisperAdapterCudaSuppression:
    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_hide_cuda_devices_when_loading_on_cpu(self, mock_whisper_cls):
        captured_env = {}

        def capture_env(*args, **kwargs):
            captured_env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")

        mock_whisper_cls.side_effect = capture_env

        adapter = FasterWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()

        assert captured_env["CUDA_VISIBLE_DEVICES"] == ""

    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_restore_cuda_visible_devices_after_cpu_load(self, mock_whisper_cls):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        try:
            adapter = FasterWhisperAdapter(model_size="base", device="cpu")
            adapter.load_model()

            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        finally:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_restore_cuda_visible_devices_after_cpu_load_failure(self, mock_whisper_cls):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        mock_whisper_cls.side_effect = RuntimeError("boom")
        try:
            adapter = FasterWhisperAdapter(model_size="base", device="cpu")
            with pytest.raises(RuntimeError):
                adapter.load_model()

            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
        finally:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_remove_env_var_if_unset_before_cpu_load(self, mock_whisper_cls):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        adapter = FasterWhisperAdapter(model_size="base", device="cpu")
        adapter.load_model()

        assert "CUDA_VISIBLE_DEVICES" not in os.environ

    @patch("voicetotext.transcription.faster_whisper_adapter.WhisperModel")
    def test_should_not_touch_cuda_env_when_loading_on_gpu(self, mock_whisper_cls):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        adapter = FasterWhisperAdapter(model_size="base", device="cuda")
        adapter.load_model()

        assert "CUDA_VISIBLE_DEVICES" not in os.environ


class TestFasterWhisperAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self):
        adapter = FasterWhisperAdapter(model_size="base")

        assert isinstance(adapter, TranscriptionEngine)
