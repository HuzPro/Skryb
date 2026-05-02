"""Tests for :class:`GraniteSpeechAdapter`.

``transformers`` and ``torch`` are patched at the module boundary via
``sys.modules`` — the same seam strategy used by the nano-parakeet
adapter tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voicetotext.core.exceptions import TranscriptionError
from voicetotext.recording.audio_segment import AudioSegment
from voicetotext.transcription.engine import TranscriptionEngine


def _mocked_modules() -> dict[str, MagicMock]:
    return {"transformers": MagicMock(), "torch": MagicMock()}


def _make_audio(sample_rate: int = 16000, samples: int = 16000) -> AudioSegment:
    return AudioSegment(
        data=np.zeros(samples, dtype=np.float32),
        sample_rate=sample_rate,
    )


def _wire_successful_transcribe(
    *,
    n_input_tokens: int = 10,
    decoded_text: str = " hello world ",
) -> tuple[MagicMock, MagicMock, dict]:
    """Configure the mock chain so a transcribe() call completes.

    Returns ``(processor, model, fake_inputs)`` so the test can make
    assertions without re-deriving them from the imports.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained.return_value
    model = AutoModelForSpeechSeq2Seq.from_pretrained.return_value
    processor.tokenizer.apply_chat_template.return_value = "<chat-prompt>"
    fake_input_ids = MagicMock()
    fake_input_ids.shape = [1, n_input_tokens]
    fake_inputs = {"input_ids": fake_input_ids}
    processor.return_value.to.return_value = fake_inputs
    processor.tokenizer.batch_decode.return_value = [decoded_text]
    return processor, model, fake_inputs


class TestGraniteSpeechAdapterIsATranscriptionEngine:
    def test_should_implement_transcription_engine_interface(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        assert isinstance(adapter, TranscriptionEngine)


class TestGraniteSpeechAdapterLoadModel:
    def test_should_report_not_loaded_before_load(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_load_processor_and_model_with_local_files_only(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/app/models"),
        )

        adapter.load_model()

        AutoProcessor.from_pretrained.assert_called_once_with(
            "ibm-granite/granite-speech-4.1-2b",
            cache_dir=str(Path("/app/models")),
            local_files_only=True,
        )
        load_kwargs = AutoModelForSpeechSeq2Seq.from_pretrained.call_args.kwargs
        load_args = AutoModelForSpeechSeq2Seq.from_pretrained.call_args.args
        assert load_args == ("ibm-granite/granite-speech-4.1-2b",)
        assert load_kwargs["cache_dir"] == str(Path("/app/models"))
        assert load_kwargs["local_files_only"] is True
        assert load_kwargs["device_map"] == "cpu"
        assert adapter.is_model_loaded() is True

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_default_to_bfloat16_dtype(self) -> None:
        # bfloat16 is Granite Speech 4.1's native training dtype. Unlike
        # nano-parakeet (which forces fp32 to dodge a kernel mismatch on
        # ROCm), Granite's bf16 path is well-trodden — keep the model
        # card's recommended dtype.
        import torch
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cuda",
            model_dir=Path("/app/models"),
        )

        adapter.load_model()

        kwargs = AutoModelForSpeechSeq2Seq.from_pretrained.call_args.kwargs
        assert kwargs["torch_dtype"] is torch.bfloat16

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_honour_explicit_dtype_override(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cuda",
            model_dir=Path("/app/models"),
            dtype="float16",
        )

        adapter.load_model()

        kwargs = AutoModelForSpeechSeq2Seq.from_pretrained.call_args.kwargs
        assert kwargs["torch_dtype"] is torch.float16


class TestGraniteSpeechAdapterTranscribeGuards:
    def test_should_raise_when_transcribing_before_load(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        with pytest.raises(TranscriptionError, match="not loaded"):
            adapter.transcribe(_make_audio())

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_raise_when_audio_sample_rate_is_not_16khz(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        audio_44k = _make_audio(sample_rate=44100, samples=44100)

        with pytest.raises(TranscriptionError, match="16000"):
            adapter.transcribe(audio_44k)


class TestGraniteSpeechAdapterTranscribe:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_build_chat_template_with_audio_token(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        processor, _, _ = _wire_successful_transcribe()

        adapter.transcribe(_make_audio())

        call = processor.tokenizer.apply_chat_template.call_args
        chat = call.args[0]
        assert call.kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        assert len(chat) == 1
        assert chat[0]["role"] == "user"
        assert "<|audio|>" in chat[0]["content"]

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_pass_unsqueezed_torch_tensor_to_processor(self) -> None:
        import torch

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cuda",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        processor, _, _ = _wire_successful_transcribe()
        audio = _make_audio()

        adapter.transcribe(audio)

        torch.from_numpy.assert_called_once_with(audio.data)
        torch.from_numpy.return_value.unsqueeze.assert_called_once_with(0)
        expected_wav = torch.from_numpy.return_value.unsqueeze.return_value
        processor.assert_called_once_with(
            "<chat-prompt>",
            expected_wav,
            device="cuda",
            return_tensors="pt",
        )
        processor.return_value.to.assert_called_once_with("cuda")

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_decode_only_tokens_after_input_ids_length(self) -> None:
        # Granite is an autoregressive LLM: ``model.generate`` returns
        # ``[input_prompt_tokens || generated_tokens]``. Decoding the
        # full output would echo the chat template back. Slice off the
        # prompt before decoding.
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        processor, model, _ = _wire_successful_transcribe(n_input_tokens=42)

        adapter.transcribe(_make_audio())

        decoded_arg = processor.tokenizer.batch_decode.call_args.args[0]
        assert decoded_arg is model.generate.return_value.__getitem__.return_value
        getitem_key = model.generate.return_value.__getitem__.call_args.args[0]
        assert getitem_key == (slice(None, None, None), slice(42, None, None))
        decode_kwargs = processor.tokenizer.batch_decode.call_args.kwargs
        assert decode_kwargs == {"skip_special_tokens": True}

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_return_stripped_text_in_transcription_result(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )
        from voicetotext.transcription.result import TranscriptionResult

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()
        _wire_successful_transcribe(decoded_text="  Hello, world!  ")

        result = adapter.transcribe(_make_audio())

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello, world!"
        assert result.language == "en"
        assert result.duration_seconds >= 0.0


class TestGraniteSpeechAdapterDispose:
    @patch.dict("sys.modules", _mocked_modules())
    def test_should_clear_model_on_dispose(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        assert adapter.is_model_loaded() is False

    @patch.dict("sys.modules", _mocked_modules())
    def test_should_release_gpu_cache_on_dispose_when_device_is_cuda(self) -> None:
        import sys

        torch_mock = MagicMock()
        sys.modules["torch"] = torch_mock

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
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

        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )
        adapter.load_model()

        adapter.dispose()

        torch_mock.cuda.empty_cache.assert_not_called()

    def test_should_not_raise_on_dispose_when_not_loaded(self) -> None:
        from voicetotext.transcription.granite_speech_adapter import (
            GraniteSpeechAdapter,
        )

        adapter = GraniteSpeechAdapter(
            repo_id="ibm-granite/granite-speech-4.1-2b",
            device="cpu",
            model_dir=Path("/tmp/models"),
        )

        adapter.dispose()
