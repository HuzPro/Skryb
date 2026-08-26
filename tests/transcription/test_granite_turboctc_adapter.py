"""Tests for :class:`GraniteTurboCtcAdapter`.

``transformers`` is patched at the module boundary via ``sys.modules``
so tests never import the real backend.
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


_REPO_ID = "ibm-granite/granite-speech-5.0-470m-turboctc"


def _make_audio(samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=16000,
    )


def _mocked_modules() -> dict:
    return {"transformers": MagicMock(), "torch": MagicMock()}


def _make_adapter(device: str = "cpu") -> TranscriptionEngine:
    from voicetotext.transcription.granite_turboctc_adapter import (
        GraniteTurboCtcAdapter,
    )

    return GraniteTurboCtcAdapter(
        repo_id=_REPO_ID,
        device=device,
        model_dir=Path("/tmp/models"),
    )


def _loaded_backend() -> tuple[MagicMock, MagicMock]:
    """The (model, processor) pair the adapter ends up holding after load.

    The model is the result of ``from_pretrained(...).to(device)``; the
    processor call chain is wired so ``processor(...)`` returns inputs
    whose ``.to(...)`` yields a real dict (``generate(**...)`` needs a
    mapping, which a bare MagicMock is not).
    """
    from transformers import GraniteSpeech5ForCTC, GraniteSpeech5Processor

    model = GraniteSpeech5ForCTC.from_pretrained.return_value.to.return_value
    processor = GraniteSpeech5Processor.from_pretrained.return_value
    processor.return_value.to.return_value = {
        "input_features": MagicMock(), "attention_mask": MagicMock(),
    }
    return model, processor


class TestGraniteTurboCtcAdapterLoadModel:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_model_from_app_cache_with_fp32_on_cpu(self) -> None:
        import torch
        from transformers import GraniteSpeech5ForCTC

        adapter = _make_adapter(device="cpu")

        adapter.load_model()

        GraniteSpeech5ForCTC.from_pretrained.assert_called_once_with(
            _REPO_ID,
            cache_dir=str(Path("/tmp/models")),
            local_files_only=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_request_bfloat16_and_sdpa_on_gpu(self) -> None:
        # The checkpoint ships bf16 weights; SDPA is the fastest attention
        # the model supports (flash-attn is explicitly unsupported by its
        # float attention bias).
        import torch
        from transformers import GraniteSpeech5ForCTC

        adapter = _make_adapter(device="cuda")

        adapter.load_model()

        kwargs = GraniteSpeech5ForCTC.from_pretrained.call_args.kwargs
        assert kwargs["dtype"] is torch.bfloat16
        assert kwargs["attn_implementation"] == "sdpa"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_move_model_to_configured_device(self) -> None:
        from transformers import GraniteSpeech5ForCTC

        adapter = _make_adapter(device="cuda")

        adapter.load_model()

        GraniteSpeech5ForCTC.from_pretrained.return_value.to.assert_called_once_with(
            "cuda",
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_processor_from_app_cache(self) -> None:
        from transformers import GraniteSpeech5Processor

        adapter = _make_adapter()

        adapter.load_model()

        GraniteSpeech5Processor.from_pretrained.assert_called_once_with(
            _REPO_ID,
            cache_dir=str(Path("/tmp/models")),
            local_files_only=True,
        )

    def test_should_report_not_loaded_before_load(self) -> None:
        adapter = _make_adapter()

        assert adapter.is_model_loaded() is False


class TestGraniteTurboCtcAdapterTranscribe:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_return_decoded_text_in_result(self) -> None:
        _, processor = _loaded_backend()
        processor.batch_decode.return_value = [" Hello, world! "]
        adapter = _make_adapter()
        adapter.load_model()

        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, world!"
        assert result.language == "en"
        assert result.transcription_seconds >= 0.0

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_audio_data_and_sample_rate_to_processor(self) -> None:
        _, processor = _loaded_backend()
        processor.batch_decode.return_value = ["text"]
        adapter = _make_adapter(device="cpu")
        adapter.load_model()
        audio = _make_audio()

        adapter.transcribe(audio)

        call = processor.call_args
        assert np.array_equal(call.args[0], audio.data)
        assert call.kwargs["sampling_rate"] == 16000
        assert call.kwargs["device"] == "cpu"

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_run_single_greedy_generate_over_processed_features(
        self,
    ) -> None:
        # Non-autoregressive CTC: one generate call per utterance, fed the
        # exact features the processor produced.
        model, processor = _loaded_backend()
        processor.batch_decode.return_value = ["text"]
        adapter = _make_adapter()
        adapter.load_model()

        adapter.transcribe(_make_audio())

        model.generate.assert_called_once_with(
            **processor.return_value.to.return_value,
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_decode_generated_sequences_without_special_tokens(
        self,
    ) -> None:
        model, processor = _loaded_backend()
        processor.batch_decode.return_value = ["text"]
        adapter = _make_adapter()
        adapter.load_model()

        adapter.transcribe(_make_audio())

        processor.batch_decode.assert_called_once_with(
            model.generate.return_value, skip_special_tokens=True,
        )

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_raise_when_transcribing_before_load(self) -> None:
        adapter = _make_adapter()

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())


class TestGraniteTurboCtcAdapterWarmup:
    """Smoke test: the inherited default warmup runs end-to-end through
    the CTC generate path so GPU kernels are compiled before the first
    user-triggered transcription."""

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_invoke_generate_during_warmup(self) -> None:
        model, processor = _loaded_backend()
        processor.batch_decode.return_value = [""]
        adapter = _make_adapter()
        adapter.load_model()

        adapter.warmup()

        assert model.generate.call_count == 1


class TestGraniteTurboCtcAdapterDispose:
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


class TestGraniteTurboCtcAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self) -> None:
        adapter = _make_adapter()

        assert isinstance(adapter, TranscriptionEngine)


class TestGraniteTurboCtcAdapterDependencyMissing:
    """When ``transformers`` is not importable,
    :meth:`GraniteTurboCtcAdapter.load_model` must raise
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
