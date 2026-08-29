"""Tests for :class:`Qwen3AsrAdapter`.

``transformers`` is patched at the module boundary via ``sys.modules``
so tests never import the real backend - the same isolation the
nano-parakeet adapter tests use.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import (
    EngineDependencyMissingError,
    TranscriptionError,
)
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine
from voicetotext.transcription.result import TranscriptionResult


_REPO_ID = "Qwen/Qwen3-ASR-1.7B-hf"
_PROMPT_LENGTH = 7


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
    )


def _mocked_modules() -> dict:
    return {"transformers": MagicMock(), "torch": MagicMock()}


def _make_adapter(device: str = "cpu") -> TranscriptionEngine:
    from voicetotext.transcription.qwen3_asr_adapter import Qwen3AsrAdapter

    return Qwen3AsrAdapter(
        repo_id=_REPO_ID,
        device=device,
        model_dir=Path("/tmp/models"),
        language="en",
    )


def _prepared_inputs() -> dict:
    input_ids = MagicMock()
    input_ids.shape = (1, _PROMPT_LENGTH)
    return {"input_ids": input_ids, "input_features": MagicMock()}


def _stub_transcription_pipeline(text: str = "text") -> dict:
    """Wire the processor/model mocks for one transcribe call.

    ``apply_transcription_request(...).to(...)`` returns a real dict so
    ``model.generate(**inputs)`` can unpack it, and ``decode`` returns
    *text*.
    """
    import transformers

    processor = transformers.AutoProcessor.from_pretrained.return_value
    model = (
        transformers.Qwen3ASRForConditionalGeneration
        .from_pretrained.return_value
    )
    inputs = _prepared_inputs()
    processor.apply_transcription_request.return_value.to.return_value = inputs
    processor.decode.return_value = text
    return {"processor": processor, "model": model, "inputs": inputs}


class TestQwen3AsrAdapterLoadModel:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_processor_and_model_from_app_models_dir(self) -> None:
        import transformers

        adapter = _make_adapter(device="cuda")

        adapter.load_model()

        transformers.AutoProcessor.from_pretrained.assert_called_once_with(
            _REPO_ID,
            cache_dir=str(Path("/tmp/models")),
            local_files_only=True,
        )
        model_kwargs = (
            transformers.Qwen3ASRForConditionalGeneration
            .from_pretrained.call_args.kwargs
        )
        assert model_kwargs["cache_dir"] == str(Path("/tmp/models"))
        assert model_kwargs["local_files_only"] is True
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_request_bfloat16_and_sdpa_on_gpu(self) -> None:
        import torch
        import transformers

        adapter = _make_adapter(device="cuda")

        adapter.load_model()

        model_kwargs = (
            transformers.Qwen3ASRForConditionalGeneration
            .from_pretrained.call_args.kwargs
        )
        assert model_kwargs["dtype"] is torch.bfloat16
        assert model_kwargs["attn_implementation"] == "sdpa"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_request_fp32_on_cpu(self) -> None:
        # The bf16 weights load fine on CPU but matmul there has no fast
        # bf16 path, so the loader requests fp32 off-GPU.
        import torch
        import transformers

        adapter = _make_adapter(device="cpu")

        adapter.load_model()

        model_kwargs = (
            transformers.Qwen3ASRForConditionalGeneration
            .from_pretrained.call_args.kwargs
        )
        assert model_kwargs["dtype"] is torch.float32

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_move_model_to_device_and_set_eval_mode(self) -> None:
        import transformers

        adapter = _make_adapter(device="cuda")

        adapter.load_model()

        model = (
            transformers.Qwen3ASRForConditionalGeneration
            .from_pretrained.return_value
        )
        model.to.assert_called_once_with("cuda")
        model.eval.assert_called_once_with()

    def test_should_report_not_loaded_before_load(self) -> None:
        adapter = _make_adapter()

        assert adapter.is_model_loaded() is False


class TestQwen3AsrAdapterTranscribe:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_return_transcribed_text_in_result(self) -> None:
        _stub_transcription_pipeline(text=" Hello, world! ")
        adapter = _make_adapter()
        adapter.load_model()

        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, world!"
        assert result.language == "en"
        assert result.transcription_seconds >= 0.0

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_numpy_audio_and_configured_language(self) -> None:
        stubs = _stub_transcription_pipeline()
        adapter = _make_adapter()
        adapter.load_model()
        audio = _make_audio()

        adapter.transcribe(audio)

        request_kwargs = (
            stubs["processor"].apply_transcription_request.call_args.kwargs
        )
        assert np.array_equal(request_kwargs["audio"], audio.data)
        assert request_kwargs["language"] == "en"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_move_inputs_to_device_and_dtype(self) -> None:
        import torch

        stubs = _stub_transcription_pipeline()
        adapter = _make_adapter(device="cuda")
        adapter.load_model()

        adapter.transcribe(_make_audio())

        request = stubs["processor"].apply_transcription_request.return_value
        request.to.assert_called_once_with("cuda", torch.bfloat16)

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_generate_greedily_with_bounded_new_tokens(self) -> None:
        stubs = _stub_transcription_pipeline()
        adapter = _make_adapter()
        adapter.load_model()

        adapter.transcribe(_make_audio())

        stubs["model"].generate.assert_called_once_with(
            input_ids=stubs["inputs"]["input_ids"],
            input_features=stubs["inputs"]["input_features"],
            max_new_tokens=256,
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_decode_only_tokens_generated_after_the_prompt(self) -> None:
        # ``generate`` returns prompt + continuation; the transcription is
        # the continuation alone, decoded in transcription-only format so
        # any ``language <NAME><asr_text>`` framing is stripped.
        stubs = _stub_transcription_pipeline()
        adapter = _make_adapter()
        adapter.load_model()

        adapter.transcribe(_make_audio())

        output_ids = stubs["model"].generate.return_value
        output_ids.__getitem__.assert_called_once_with(
            (0, slice(_PROMPT_LENGTH, None)),
        )
        stubs["processor"].decode.assert_called_once_with(
            output_ids.__getitem__.return_value,
            return_format="transcription_only",
        )

    def test_should_raise_when_transcribing_before_load(self) -> None:
        adapter = _make_adapter()

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())


class TestQwen3AsrAdapterWarmup:
    """Smoke test: the inherited default warmup runs end-to-end through
    the generate path so GPU kernels are compiled before the first
    user-triggered transcription."""

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_invoke_generate_during_warmup(self) -> None:
        stubs = _stub_transcription_pipeline(text="")
        adapter = _make_adapter()
        adapter.load_model()

        adapter.warmup()

        assert stubs["model"].generate.call_count == 1


class TestQwen3AsrAdapterDispose:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_clear_model_on_dispose(self) -> None:
        adapter = _make_adapter()
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_release_gpu_cache_on_dispose(self) -> None:
        import sys
        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock

        adapter = _make_adapter(device="cuda")
        adapter.load_model()

        adapter.dispose()

        torch_mock.cuda.empty_cache.assert_called_once()

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_skip_gpu_cache_release_on_cpu_dispose(self) -> None:
        import sys
        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock

        adapter = _make_adapter(device="cpu")
        adapter.load_model()

        adapter.dispose()

        torch_mock.cuda.empty_cache.assert_not_called()

    def test_should_not_raise_on_dispose_when_not_loaded(self) -> None:
        adapter = _make_adapter()

        adapter.dispose()


class TestQwen3AsrAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self) -> None:
        adapter = _make_adapter()

        assert isinstance(adapter, TranscriptionEngine)


class TestQwen3AsrAdapterDependencyMissing:
    """When ``transformers`` is not installed,
    :meth:`Qwen3AsrAdapter.load_model` must raise
    :class:`EngineDependencyMissingError` with an actionable install
    hint instead of leaking a bare ``ImportError``.
    """

    def test_should_raise_engine_dependency_missing_when_backend_not_installed(
        self,
    ) -> None:
        adapter = _make_adapter()

        with patch.dict(
            "sys.modules", {"transformers": None, "torch": MagicMock()}
        ):
            with pytest.raises(EngineDependencyMissingError) as exc_info:
                adapter.load_model()

        message = str(exc_info.value).lower()
        assert "transformers" in message
        assert "pip install" in message
