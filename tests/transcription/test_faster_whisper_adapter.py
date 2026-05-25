"""Tests for :class:`FasterWhisperAdapter`.

Mocks ``faster_whisper`` at the ``sys.modules`` boundary the same way
:mod:`tests.transcription.test_transformers_whisper_adapter` mocks
``transformers`` and ``torch``.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    TranscriptionError,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.faster_whisper_adapter import FasterWhisperAdapter
from voicetotext.transcription.result import TranscriptionResult


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
    )


def _segments(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=t, start=0.0, end=0.0) for t in texts]


def _mocked_modules() -> dict:
    return {"faster_whisper": MagicMock()}


def _make_adapter(
    *,
    model_name: str = "base",
    device: str = "cuda",
    repo_id: str = "Systran/faster-whisper-base",
    language: str = "en",
    beam_size: int = 5,
    model_dir: Path | None = None,
) -> FasterWhisperAdapter:
    return FasterWhisperAdapter(
        model_name=model_name,
        repo_id=repo_id,
        device=device,
        language=language,
        beam_size=beam_size,
        model_dir=model_dir or Path("/tmp/models"),
    )


class TestFasterWhisperAdapterLoadModel:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_model_with_float16_on_cuda(self):
        from faster_whisper import WhisperModel

        adapter = _make_adapter(device="cuda")

        adapter.load_model()

        WhisperModel.assert_called_once()
        kwargs = WhisperModel.call_args.kwargs
        assert kwargs["device"] == "cuda"
        assert kwargs["compute_type"] == "float16"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_model_with_int8_on_cpu(self):
        from faster_whisper import WhisperModel

        adapter = _make_adapter(device="cpu")

        adapter.load_model()

        kwargs = WhisperModel.call_args.kwargs
        assert kwargs["device"] == "cpu"
        assert kwargs["compute_type"] == "int8"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_repo_id_as_first_positional(self):
        from faster_whisper import WhisperModel

        adapter = _make_adapter(
            model_name="large-v3-turbo",
            repo_id="Systran/faster-whisper-large-v3-turbo",
        )

        adapter.load_model()

        assert WhisperModel.call_args.args[0] == "Systran/faster-whisper-large-v3-turbo"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_model_dir_as_download_root(self, tmp_path):
        from faster_whisper import WhisperModel

        adapter = _make_adapter(model_dir=tmp_path)

        adapter.load_model()

        kwargs = WhisperModel.call_args.kwargs
        assert Path(kwargs["download_root"]) == tmp_path

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_local_files_only_true(self):
        from faster_whisper import WhisperModel

        adapter = _make_adapter()

        adapter.load_model()

        kwargs = WhisperModel.call_args.kwargs
        assert kwargs["local_files_only"] is True

    def test_should_implement_transcription_engine_interface(self):
        adapter = _make_adapter()

        assert isinstance(adapter, TranscriptionEngine)

    def test_should_report_not_loaded_before_load(self):
        adapter = _make_adapter()

        assert adapter.is_model_loaded() is False


class TestFasterWhisperAdapterTranscribe:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_raise_when_transcribing_before_load(self):
        adapter = _make_adapter()

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_join_segments_into_single_text(self):
        from faster_whisper import WhisperModel

        info = SimpleNamespace(language="en", language_probability=0.95)
        WhisperModel.return_value.transcribe.return_value = (
            _segments("hello", " world"), info,
        )
        adapter = _make_adapter()
        adapter.load_model()

        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "hello world"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_language_and_beam_size_to_transcribe(self):
        from faster_whisper import WhisperModel

        info = SimpleNamespace(language="en", language_probability=0.95)
        WhisperModel.return_value.transcribe.return_value = (_segments(""), info)
        adapter = _make_adapter(language="en", beam_size=3)
        adapter.load_model()

        adapter.transcribe(_make_audio())

        call = WhisperModel.return_value.transcribe.call_args
        assert call.kwargs["language"] == "en"
        assert call.kwargs["beam_size"] == 3

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_carry_language_metadata_into_result(self):
        from faster_whisper import WhisperModel

        info = SimpleNamespace(language="es", language_probability=0.87)
        WhisperModel.return_value.transcribe.return_value = (_segments("hola"), info)
        adapter = _make_adapter()
        adapter.load_model()

        result = adapter.transcribe(_make_audio())

        assert result.language == "es"
        assert result.language_probability == pytest.approx(0.87)


class TestFasterWhisperAdapterWarmup:
    """Smoke test: the inherited default warmup runs end-to-end through
    faster-whisper's transcribe path so CTranslate2 kernels are
    compiled before the first user-triggered transcription."""

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_invoke_model_transcribe_during_warmup(self):
        from faster_whisper import WhisperModel

        info = SimpleNamespace(language="en", language_probability=0.0)
        WhisperModel.return_value.transcribe.return_value = (_segments(""), info)
        adapter = _make_adapter()
        adapter.load_model()

        adapter.warmup()

        assert WhisperModel.return_value.transcribe.call_count == 1


class TestFasterWhisperAdapterDependencyMissing:
    """When the ``faster_whisper`` backend is not installed,
    :meth:`FasterWhisperAdapter.load_model` must raise
    :class:`EngineDependencyMissingError` with an actionable install
    hint so the tray can surface "pip install faster-whisper" to the
    user instead of leaking a bare ``ImportError``.
    """

    def test_should_raise_engine_dependency_missing_when_backend_not_installed(self):
        adapter = _make_adapter()

        # ``sys.modules[name] = None`` is the canonical way to make
        # ``import name`` raise ``ImportError`` regardless of whether
        # the package is installed in the test environment.
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with pytest.raises(EngineDependencyMissingError) as exc_info:
                adapter.load_model()

        message = str(exc_info.value).lower()
        assert "faster-whisper" in message
        assert "pip install" in message


class TestFasterWhisperAdapterDispose:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_clear_model_on_dispose(self):
        adapter = _make_adapter()
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    def test_should_not_raise_on_dispose_when_not_loaded(self):
        adapter = _make_adapter()

        adapter.dispose()
