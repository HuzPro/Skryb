"""Tests for :class:`NanoParakeetAdapter`.

``nano_parakeet`` is patched at the module boundary via
``sys.modules`` — the same seam strategy used by the transformers
adapter tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
    )


def _mocked_modules() -> dict:
    return {"nano_parakeet": MagicMock()}


class TestNanoParakeetAdapterLoadModel:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_model_via_from_pretrained(self) -> None:
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        adapter.load_model()

        from_pretrained.assert_called_once_with(
            model_name="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
        )
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_cuda_device_to_from_pretrained(self) -> None:
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v3",
            device="cuda",
            model_dir=Path("/tmp/models"),
        )

        adapter.load_model()

        from_pretrained.assert_called_once_with(
            model_name="nvidia/parakeet-tdt-0.6b-v3",
            device="cuda",
        )

    def test_should_report_not_loaded_before_load(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        assert adapter.is_model_loaded() is False


class TestNanoParakeetAdapterTranscribe:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_return_transcribed_text_in_result(self) -> None:
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        model = from_pretrained.return_value
        model.transcribe.return_value = " Hello, world! "

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, world!"
        assert result.duration_seconds >= 0.0

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_numpy_audio_data_to_model(self) -> None:
        from nano_parakeet import from_pretrained

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        model = from_pretrained.return_value
        model.transcribe.return_value = "text"

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        audio = _make_audio()

        adapter.transcribe(audio)

        passed_audio = model.transcribe.call_args[0][0]
        assert np.array_equal(passed_audio, audio.data)

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_raise_when_transcribing_before_load(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())


class TestNanoParakeetAdapterDispose:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_clear_model_on_dispose(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_release_gpu_cache_on_dispose(self) -> None:
        import sys
        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cuda",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        torch_mock.cuda.empty_cache.assert_called_once()

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_skip_gpu_cache_release_on_cpu_dispose(self) -> None:
        import sys
        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock

        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        torch_mock.cuda.empty_cache.assert_not_called()

    def test_should_not_raise_on_dispose_when_not_loaded(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        adapter.dispose()


class TestNanoParakeetAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self) -> None:
        from voicetotext.transcription.nano_parakeet_adapter import (
            NanoParakeetAdapter,
        )

        adapter = NanoParakeetAdapter(
            repo_id="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        assert isinstance(adapter, TranscriptionEngine)
